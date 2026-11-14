from collections.abc import AsyncGenerator, AsyncIterator, Collection, Generator, Iterator
from dataclasses import dataclass
from typing import Literal, overload
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
from pregel_redux.configs import ThreadConfig, parse_run_target, parse_state_address, read_config
from pregel_redux.dependencies import Depends, Scope
from pregel_redux.graph.adapters import iterate_in_new_loop, run_in_new_loop, stream_run
from pregel_redux.graph.compiled import CompiledGraph
from pregel_redux.graph.threads import ThreadJournal, ThreadScope, find_graph, infer_writer
from pregel_redux.interrupts import Finished, Interrupted, ResumeInput
from pregel_redux.pregel import (
    Position,
    PullTask,
    Task,
    TaskWrites,
    apply_command,
    apply_writes,
    plan_tasks,
    read_command_writes,
)
from pregel_redux.shutdown import RunAborted, ShutdownController, ShutdownSignal, read_shutdown
from pregel_redux.snapshots import StateSnapshot
from pregel_redux.streams import (
    CheckpointPart,
    CustomPart,
    InterruptsPart,
    Stream,
    StreamMode,
    StreamPart,
    TaskResultPart,
    TaskStartPart,
    UpdatesPart,
    ValuesPart,
    parse_stream_modes,
)
from pregel_redux.task_refs import refer_task


@dataclass(frozen=True)
class RunStart:
    position: Position
    journal: ThreadJournal
    resume: ResumeInput
    resumed: bool


@dataclass(frozen=True)
class PersistentGraph[StateT, InputT, OutputT]:
    graph: CompiledGraph[StateT, InputT, OutputT]
    checkpointer: Checkpointer

    async def start(
        self,
        checkpointer: AsyncCheckpointer,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
    ) -> RunStart:
        target = parse_run_target(config)
        thread_id = target.thread_id
        scope = ThreadScope(self.graph, checkpointer, thread_id, "")
        match input:
            case Command() as command:
                answers = command.resume is not None or bool(command.resume_by_interrupt_id)
            case _:
                answers = False
        opened = await scope.open_run(target.fork_from, answers=answers)
        match input:
            case None | Command():
                if opened is None:
                    raise LookupError(f"thread {thread_id!r} has no checkpoint, so the run needs an input")
                position, journal = opened.position, opened.journal
                resume = await journal.read_resume(input)
                writes = (
                    None if input is None else read_command_writes(self.graph.nodes.keys(), self.graph.channels, input)
                )
                if writes is not None and (writes.fields or writes.branches or writes.sends):
                    await journal.record_command(writes)
                    position = apply_command(self.graph.trigger_keys, self.graph.triggers, position, writes)
                return RunStart(position, journal, resume, resumed=True)
            case _ if opened is None:
                position = self.graph.feed(self.graph.open({}), extract_writes(input))
                journal = await scope.begin(position, -1, None)
            case _:
                position = self.graph.feed(opened.position, extract_writes(input))
                checkpoint = opened.journal.saved.checkpoint
                journal = await scope.begin(position, checkpoint.step + 1, checkpoint.id)
        return RunStart(position, journal, ResumeInput([], {}), resumed=False)

    async def run(
        self,
        checkpointer: AsyncCheckpointer,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        step_limit: int | None,
        stream: Stream,
        signal: ShutdownSignal | None,
    ) -> Finished[OutputT] | Interrupted:
        shutdown = ShutdownController().signal if signal is None else signal
        if shutdown.aborted:
            raise RunAborted("the shutdown signal of the run is aborted")
        return await self.run_thread(checkpointer, input, config, step_limit, stream, shutdown)

    async def run_thread(
        self,
        checkpointer: AsyncCheckpointer,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        step_limit: int | None,
        stream: Stream,
        shutdown: ShutdownSignal,
    ) -> Finished[OutputT] | Interrupted:
        start = await self.start(checkpointer, input, config)
        provided: dict[Depends, object] = {
            **{Depends(read_config, scope=scope): config for scope in Scope},
            **{Depends(read_shutdown, scope=scope): shutdown for scope in Scope},
        }
        return await self.graph.run(
            start.position, step_limit, start.journal, start.resume, stream, provided, shutdown, resumed=start.resumed
        )

    async def ainvoke(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Finished[OutputT] | Interrupted:
        stream = Stream(frozenset(), subgraphs=False, emit=lambda part: None)
        return await self.run(self.checkpointer, input, config, step_limit, stream, signal)

    def invoke(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Finished[OutputT] | Interrupted:
        self.graph.reject_async_nodes("ainvoke")
        stream = Stream(frozenset(), subgraphs=False, emit=lambda part: None)
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return run_in_new_loop(lambda: self.run(checkpointer, input, config, step_limit, stream, signal))

    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.values],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[ValuesPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.updates],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[UpdatesPart | InterruptsPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.custom],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[CustomPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.tasks],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[TaskStartPart | TaskResultPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.checkpoints],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[CheckpointPart]: ...
    @overload
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[StreamPart]: ...
    def astream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: StreamMode | Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> AsyncGenerator[StreamPart]:
        modes = parse_stream_modes(stream_mode)
        return stream_run(
            lambda emit: self.run(
                self.checkpointer,
                input,
                config,
                step_limit,
                Stream(modes, subgraphs=subgraphs, emit=emit),
                signal,
            )
        )

    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.values],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[ValuesPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.updates],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[UpdatesPart | InterruptsPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.custom],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[CustomPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.tasks],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[TaskStartPart | TaskResultPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Literal[StreamMode.checkpoints],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[CheckpointPart]: ...
    @overload
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[StreamPart]: ...
    def stream(
        self,
        input: InputT | Command[str] | None,
        config: ThreadConfig,
        *,
        stream_mode: StreamMode | Collection[StreamMode],
        subgraphs: bool = False,
        step_limit: int | None = None,
        signal: ShutdownSignal | None = None,
    ) -> Generator[StreamPart]:
        self.graph.reject_async_nodes("astream")
        modes = parse_stream_modes(stream_mode)
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return iterate_in_new_loop(
            lambda: stream_run(
                lambda emit: self.run(
                    checkpointer,
                    input,
                    config,
                    step_limit,
                    Stream(modes, subgraphs=subgraphs, emit=emit),
                    signal,
                )
            )
        )

    async def read_state(
        self, checkpointer: AsyncCheckpointer, config: ThreadConfig, *, subgraphs: bool
    ) -> StateSnapshot:
        address = parse_state_address(config)
        graph = find_graph(self.graph, address.checkpoint_ns)
        scope = ThreadScope(graph, checkpointer, address.thread_id, address.checkpoint_ns)
        saved = await checkpointer.aget_tuple(address.thread_id, address.checkpoint_ns, address.checkpoint_id)
        if saved is None:
            raise LookupError(f"thread {address.thread_id!r} has no such checkpoint")
        return await scope.read_snapshot(saved, subgraphs=subgraphs)

    async def aget_state(self, config: ThreadConfig, *, subgraphs: bool = False) -> StateSnapshot:
        return await self.read_state(self.checkpointer, config, subgraphs=subgraphs)

    def get_state(self, config: ThreadConfig, *, subgraphs: bool = False) -> StateSnapshot:
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return run_in_new_loop(lambda: self.read_state(checkpointer, config, subgraphs=subgraphs))

    async def read_state_history(
        self, checkpointer: AsyncCheckpointer, config: ThreadConfig, before: UUID | None, limit: int | None
    ) -> AsyncIterator[StateSnapshot]:
        address = parse_state_address(config)
        thread_id, checkpoint_ns = address.thread_id, address.checkpoint_ns
        if limit is not None and limit < 0:
            raise ValueError(f"limit is {limit}, and the number of checkpoints to read cannot be negative")
        scope = ThreadScope(find_graph(self.graph, checkpoint_ns), checkpointer, thread_id, checkpoint_ns)
        if address.checkpoint_id is not None:
            saved = await checkpointer.aget_tuple(thread_id, checkpoint_ns, address.checkpoint_id)
            if saved is not None and (before is None or saved.checkpoint.id < before) and limit != 0:
                yield await scope.read_snapshot(saved, subgraphs=False)
            return
        async for saved in checkpointer.alist(thread_id, checkpoint_ns, before=before, limit=limit):
            yield await scope.read_snapshot(saved, subgraphs=False)

    def aget_state_history(
        self, config: ThreadConfig, *, before: UUID | None = None, limit: int | None = None
    ) -> AsyncIterator[StateSnapshot]:
        return self.read_state_history(self.checkpointer, config, before, limit)

    def get_state_history(
        self, config: ThreadConfig, *, before: UUID | None = None, limit: int | None = None
    ) -> Iterator[StateSnapshot]:
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return iterate_in_new_loop(lambda: self.read_state_history(checkpointer, config, before, limit))

    async def write_update(
        self, checkpointer: AsyncCheckpointer, config: ThreadConfig, values: object, as_node: str | None
    ) -> UUID:
        address = parse_state_address(config)
        graph = find_graph(self.graph, address.checkpoint_ns)
        scope = ThreadScope(graph, checkpointer, address.thread_id, address.checkpoint_ns)
        loaded = await scope.find(address.checkpoint_id)
        finished: list[tuple[Task, TaskWrites]] = []
        if loaded is None:
            node = infer_writer(graph, {}) if as_node is None else as_node
            position, step, parent_id, newest_id = graph.open({}), -1, None, None
        else:
            checkpoint = loaded.saved.checkpoint
            node = infer_writer(graph, checkpoint.versions_seen) if as_node is None else as_node
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

    async def aupdate_state(self, config: ThreadConfig, values: object, *, as_node: str | None = None) -> UUID:
        return await self.write_update(self.checkpointer, config, values, as_node)

    def update_state(self, config: ThreadConfig, values: object, *, as_node: str | None = None) -> UUID:
        checkpointer = ThreadedCheckpointer(self.checkpointer)
        return run_in_new_loop(lambda: self.write_update(checkpointer, config, values, as_node))
