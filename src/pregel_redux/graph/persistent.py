from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from dataclasses import dataclass
from functools import cached_property
from uuid import UUID

from pregel_redux.channels import (
    extract_writes,
)
from pregel_redux.checkpoints import (
    AsyncCheckpointer,
    Checkpointer,
    CheckpointSource,
    ThreadedCheckpointer,
)
from pregel_redux.commands import Command
from pregel_redux.configs import Namespace, RunTarget, StateAddress, parse_run_target, parse_state_address
from pregel_redux.graph.adapters import iterate_in_new_loop, run_in_new_loop, stream_run
from pregel_redux.graph.compiled import Breakpoints, CompiledGraph, Continuation, FreshStart, RunOptions, Start
from pregel_redux.graph.threads import ThreadJournal, ThreadScope, find_graph, infer_writer
from pregel_redux.messages import Message
from pregel_redux.pregel import (
    PullTask,
    Task,
    TaskWrites,
    apply_command,
    apply_writes,
    plan_tasks,
    read_command_writes,
)
from pregel_redux.results import Finished, Stopped, Waiting
from pregel_redux.shutdown import RunAborted, ShutdownController, ShutdownSignal
from pregel_redux.snapshots import StateSnapshot
from pregel_redux.streams import (
    ChildPart,
    Stream,
    StreamPart,
)
from pregel_redux.task_refs import refer_task


@dataclass(frozen=True)
class PersistentGraph[StateT, InputT, OutputT]:
    graph: CompiledGraph[StateT, InputT, OutputT]
    checkpointer: Checkpointer
    breakpoints: Breakpoints

    @cached_property
    def storage(self) -> AsyncCheckpointer:
        return ThreadedCheckpointer(self.checkpointer)

    async def start(
        self,
        input: InputT | Command[str] | Message | None,
        target: RunTarget,
    ) -> Start:
        thread_id = target.thread_id
        scope = ThreadScope(self.graph, self.storage, thread_id, ())
        match input:
            case Message():
                return await scope.receive(input, target.fork_from)
            case _:
                pass
        opened = await scope.open_run(target.fork_from)
        match input:
            case None | Command():
                if opened is None:
                    raise LookupError(f"thread {thread_id!r} has no checkpoint, so the run needs an input")
                position, journal = opened.position, opened.journal
                writes = (
                    None if input is None else read_command_writes(self.graph.nodes.keys(), self.graph.channels, input)
                )
                if writes is not None and (writes.fields or writes.branches or writes.sends):
                    self.graph.reject_sends_to_receivers(writes.sends)
                    await journal.record_command(writes)
                    position = apply_command(self.graph.trigger_keys, self.graph.triggers, position, writes)
                return Continuation(position, journal)
            case _ if opened is None:
                position = self.graph.feed(self.graph.open({}), extract_writes(input))
                journal = await scope.begin(position, -1, None)
            case _:
                position = self.graph.feed(opened.position, extract_writes(input))
                checkpoint = opened.journal.saved.checkpoint
                journal = await scope.begin(position, checkpoint.step + 1, checkpoint.id)
        return FreshStart(position, journal)

    async def run(
        self,
        input: InputT | Command[str] | Message | None,
        target: RunTarget,
        step_limit: int | None,
        stream: Stream,
        signal: ShutdownSignal | None,
    ) -> Finished[OutputT] | Waiting | Stopped:
        shutdown = ShutdownController().signal if signal is None else signal
        if shutdown.aborted:
            raise RunAborted("the shutdown signal of the run is aborted")
        return await self.run_thread(input, target, step_limit, stream, shutdown)

    async def run_thread(
        self,
        input: InputT | Command[str] | Message | None,
        target: RunTarget,
        step_limit: int | None,
        stream: Stream,
        shutdown: ShutdownSignal,
    ) -> Finished[OutputT] | Waiting | Stopped:
        options = RunOptions(
            step_limit=step_limit,
            stream=stream,
            shutdown=shutdown,
            provided={RunTarget: target, ShutdownSignal: shutdown},
            breakpoints=self.breakpoints,
        )
        return await self.graph.run(await self.start(input, target), options)

    async def ainvoke(
        self,
        input: InputT | Command[str] | Message | None,
        thread_id: str,
        *,
        fork_from: UUID | None = None,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Finished[OutputT] | Waiting | Stopped:
        stream = Stream(lambda part: None)
        target = parse_run_target(thread_id, fork_from)
        return await self.run(input, target, step_limit, stream, signal)

    def invoke(
        self,
        input: InputT | Command[str] | Message | None,
        thread_id: str,
        *,
        fork_from: UUID | None = None,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Finished[OutputT] | Waiting | Stopped:
        self.graph.reject_async_nodes("ainvoke")
        stream = Stream(lambda part: None)
        target = parse_run_target(thread_id, fork_from)
        return run_in_new_loop(lambda: self.run(input, target, step_limit, stream, signal))

    def astream(
        self,
        input: InputT | Command[str] | Message | None,
        thread_id: str,
        *,
        fork_from: UUID | None = None,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[StreamPart | ChildPart]:
        target = parse_run_target(thread_id, fork_from)
        return stream_run(
            lambda emit: self.run(
                input,
                target,
                step_limit,
                Stream(emit),
                signal,
            )
        )

    def stream(
        self,
        input: InputT | Command[str] | Message | None,
        thread_id: str,
        *,
        fork_from: UUID | None = None,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[StreamPart | ChildPart]:
        self.graph.reject_async_nodes("astream")
        target = parse_run_target(thread_id, fork_from)
        return iterate_in_new_loop(
            lambda: stream_run(
                lambda emit: self.run(
                    input,
                    target,
                    step_limit,
                    Stream(emit),
                    signal,
                )
            )
        )

    async def read_state(self, address: StateAddress, *, subgraphs: bool) -> StateSnapshot:
        graph = find_graph(self.graph, address.checkpoint_ns)
        scope = ThreadScope(graph, self.storage, address.thread_id, address.checkpoint_ns)
        saved = await self.storage.aget_tuple(address.thread_id, scope.stored_ns, address.checkpoint_id)
        if saved is None:
            raise LookupError(f"thread {address.thread_id!r} has no such checkpoint")
        return await scope.read_snapshot(saved, subgraphs=subgraphs)

    async def aget_state(
        self,
        thread_id: str,
        *,
        checkpoint_ns: Namespace = (),
        checkpoint_id: UUID | None = None,
        subgraphs: bool = False,
    ) -> StateSnapshot:
        address = parse_state_address(thread_id, checkpoint_ns, checkpoint_id)
        return await self.read_state(address, subgraphs=subgraphs)

    def get_state(
        self,
        thread_id: str,
        *,
        checkpoint_ns: Namespace = (),
        checkpoint_id: UUID | None = None,
        subgraphs: bool = False,
    ) -> StateSnapshot:
        address = parse_state_address(thread_id, checkpoint_ns, checkpoint_id)
        return run_in_new_loop(lambda: self.read_state(address, subgraphs=subgraphs))

    async def read_state_history(
        self, address: StateAddress, before: UUID | None, limit: int | None
    ) -> AsyncIterator[StateSnapshot]:
        thread_id, checkpoint_ns = address.thread_id, address.checkpoint_ns
        if limit is not None and limit < 0:
            raise ValueError(f"limit is {limit}, and the number of checkpoints to read cannot be negative")
        scope = ThreadScope(find_graph(self.graph, checkpoint_ns), self.storage, thread_id, checkpoint_ns)
        async for saved in self.storage.alist(thread_id, scope.stored_ns, before=before, limit=limit):
            yield await scope.read_snapshot(saved, subgraphs=False)

    def aget_state_history(
        self, thread_id: str, *, checkpoint_ns: Namespace = (), before: UUID | None = None, limit: int | None = None
    ) -> AsyncIterator[StateSnapshot]:
        address = parse_state_address(thread_id, checkpoint_ns, None)
        return self.read_state_history(address, before, limit)

    def get_state_history(
        self, thread_id: str, *, checkpoint_ns: Namespace = (), before: UUID | None = None, limit: int | None = None
    ) -> Iterator[StateSnapshot]:
        address = parse_state_address(thread_id, checkpoint_ns, None)
        return iterate_in_new_loop(lambda: self.read_state_history(address, before, limit))

    async def write_update(self, address: StateAddress, values: object, as_node: str | None) -> UUID:
        graph = find_graph(self.graph, address.checkpoint_ns)
        scope = ThreadScope(graph, self.storage, address.thread_id, address.checkpoint_ns)
        loaded = await scope.find(address.checkpoint_id)
        finished: list[tuple[Task, TaskWrites]] = []
        if loaded is None:
            node = infer_writer(graph, {}, ()) if as_node is None else as_node
            position, step, parent_id, newest_id = graph.open({}), -1, None, None
        else:
            checkpoint = loaded.saved.checkpoint
            node = (
                infer_writer(graph, checkpoint.versions_seen, await scope.list_nodes_of_the_last_step(checkpoint))
                if as_node is None
                else as_node
            )
            position = scope.read_position(loaded.saved)
            if address.checkpoint_id is None:
                journal = ThreadJournal(scope, loaded.saved, loaded.newest_id)
                finished = [
                    (task, writes)
                    for task in plan_tasks(position, graph.triggers)
                    if (writes := journal.find_writes(refer_task(task))) is not None
                ]
            step, parent_id, newest_id = checkpoint.step, checkpoint.id, loaded.newest_id
        if node not in graph.nodes:
            raise ValueError(f"as_node is {node!r}, which is not a node of this graph")
        updated = apply_writes(
            position,
            [*(task for task, _ in finished), PullTask(node, ())],
            [*(writes for _, writes in finished), graph.read_writes(node, values)],
            [],
            graph.trigger_keys,
        )
        saved = await scope.put(updated, step + 1, CheckpointSource.update, parent_id, newest_id)
        return saved.id

    async def aupdate_state(
        self,
        thread_id: str,
        values: object,
        *,
        as_node: str | None = None,
        checkpoint_ns: Namespace = (),
        checkpoint_id: UUID | None = None,
    ) -> UUID:
        address = parse_state_address(thread_id, checkpoint_ns, checkpoint_id)
        return await self.write_update(address, values, as_node)

    def update_state(
        self,
        thread_id: str,
        values: object,
        *,
        as_node: str | None = None,
        checkpoint_ns: Namespace = (),
        checkpoint_id: UUID | None = None,
    ) -> UUID:
        address = parse_state_address(thread_id, checkpoint_ns, checkpoint_id)
        return run_in_new_loop(lambda: self.write_update(address, values, as_node))
