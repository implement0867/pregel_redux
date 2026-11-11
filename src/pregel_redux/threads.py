from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from typing import Never, assert_never
from uuid import UUID

from pydantic_core import to_json

from pregel_redux.channels import apply_writes, extract_writes, reject_unknown_channels
from pregel_redux.checkpoints import (
    Checkpoint,
    Checkpointer,
    CheckpointSource,
    CompletedTask,
    EdgeTask,
    InterruptedTask,
    SavedCheckpoint,
    SendTask,
    TaskAddress,
    TaskStateKind,
    dump_value,
)
from pregel_redux.dependencies import CallPlan
from pregel_redux.graph import (
    CompiledGraph,
    Position,
    PullTask,
    PushTask,
    RunResult,
    Task,
    TaskOutput,
    plan_tasks,
    run_coroutine,
)
from pregel_redux.interrupts import Answer, Interrupt, InterruptId

type AnyGraph = CompiledGraph[object, Never, object]


def reject_unknown_nodes(graph: AnyGraph, names: Iterable[str]) -> None:
    unknown = sorted(set(names) - graph.nodes.keys())
    if unknown:
        raise ValueError(f"the checkpoint refers to {unknown!r}, which are not nodes of this graph")


def settle_output(graph: AnyGraph, output: TaskOutput) -> TaskOutput:
    reject_unknown_channels(graph.channels, output.writes)
    return TaskOutput(
        {name: graph.channels[name].parse_update(value) for name, value in output.writes.items()},
        output.targets,
        tuple(PushTask(push.node, graph.nodes[push.node].build_send_input(push.payload)) for push in output.pushes),
    )


def encode_push(graph: AnyGraph, push: PushTask) -> SendTask:
    node = graph.nodes[push.node]
    node_input = dump_value(node.input_adapter, node.build_send_input(push.payload), f"a send to node {push.node!r}")
    return SendTask(node=push.node, node_input=node_input)


def decode_push(graph: AnyGraph, send: SendTask) -> PushTask:
    return PushTask(send.node, graph.nodes[send.node].input_adapter.validate_json(send.node_input, by_name=True))


def encode_task(graph: AnyGraph, task: Task) -> EdgeTask | SendTask:
    match task:
        case PullTask():
            return EdgeTask(node=task.node)
        case PushTask():
            return encode_push(graph, task)
        case _:
            assert_never(task)


def decode_task(graph: AnyGraph, task: EdgeTask | SendTask) -> Task:
    match task:
        case EdgeTask():
            return PullTask(task.node)
        case SendTask():
            return decode_push(graph, task)
        case _:
            assert_never(task)


def encode_output(graph: AnyGraph, output: TaskOutput) -> CompletedTask:
    return CompletedTask(
        writes={
            name: dump_value(graph.channels[name].update_adapter, value, f"a write to channel {name!r}")
            for name, value in output.writes.items()
        },
        targets=tuple(sorted(output.targets)),
        sends=tuple(encode_push(graph, push) for push in output.pushes),
    )


def decode_task_state(graph: AnyGraph, state: CompletedTask | InterruptedTask) -> TaskOutput | InterruptedTask:
    match state:
        case CompletedTask():
            reject_unknown_channels(graph.channels, state.writes)
            reject_unknown_nodes(graph, [*state.targets, *(send.node for send in state.sends)])
            return TaskOutput(
                {
                    name: graph.channels[name].update_adapter.validate_json(data, by_name=True)
                    for name, data in state.writes.items()
                },
                frozenset(state.targets),
                tuple(decode_push(graph, send) for send in state.sends),
            )
        case InterruptedTask():
            return state
        case _:
            assert_never(state)


def encode_position(graph: AnyGraph, position: Position, thread_id: str) -> Checkpoint:
    return Checkpoint(
        thread_id=thread_id,
        caller=position.caller,
        id=position.checkpoint_id,
        parent_id=position.parent_id,
        created_at=datetime.now(UTC),
        step=position.step,
        source=position.source,
        values={
            name: dump_value(graph.channels[name].adapter, value, f"channel {name!r}")
            for name, value in position.values.items()
        },
        tasks=tuple(encode_task(graph, task) for task in position.tasks),
    )


def decode_checkpoint(graph: AnyGraph, saved: SavedCheckpoint) -> Position:
    checkpoint = saved.checkpoint
    reject_unknown_channels(graph.channels, checkpoint.values)
    reject_unknown_nodes(graph, (task.node for task in checkpoint.tasks))
    return Position(
        checkpoint_id=checkpoint.id,
        parent_id=checkpoint.parent_id,
        caller=checkpoint.caller,
        step=checkpoint.step,
        source=checkpoint.source,
        values={
            name: graph.channels[name].adapter.validate_json(data, by_name=True)
            for name, data in checkpoint.values.items()
        },
        tasks=tuple(decode_task(graph, task) for task in checkpoint.tasks),
        progress={index: decode_task_state(graph, state) for index, state in saved.task_states.items()},
    )


def close_step(graph: AnyGraph, source: Position, as_node: str, update: object) -> list[TaskOutput]:
    if as_node not in graph.nodes:
        raise ValueError(f"{as_node!r} is not a node of this graph")
    finished: list[TaskOutput] = []
    unfinished: set[str] = set()
    for index, task in enumerate(source.tasks):
        recorded = source.progress.get(index)
        match recorded:
            case _ if task.node == as_node:
                pass
            case TaskOutput():
                finished.append(recorded)
            case InterruptedTask() | None:
                unfinished.add(task.node)
            case _:
                assert_never(recorded)
    if unfinished:
        raise ValueError(
            f"the tasks of {sorted(unfinished)!r} have not finished, so resume them or update without as_node"
        )
    return [*finished, settle_output(graph, graph.read_task_output(update, graph.nodes[as_node]))]


@dataclass(frozen=True)
class PendingInterrupt:
    interrupt: Interrupt
    answers: tuple[str, ...]


@dataclass(frozen=True)
class ThreadJournal:
    checkpointer: Checkpointer
    thread_id: str

    async def save(self, graph: AnyGraph, position: Position) -> None:
        await self.checkpointer.put_checkpoint(encode_position(graph, position, self.thread_id))

    async def record_output(self, graph: AnyGraph, task: TaskAddress, output: TaskOutput) -> TaskOutput:
        settled = settle_output(graph, output)
        await self.checkpointer.put_task_state(self.thread_id, task, encode_output(graph, settled))
        return settled

    async def record_interrupt(self, task: TaskAddress, state: InterruptedTask) -> None:
        await self.checkpointer.put_task_state(self.thread_id, task, state)

    async def load_latest(self, graph: AnyGraph, caller: TaskAddress | None) -> Position | None:
        saved = await self.read_latest(caller)
        return None if saved is None else decode_checkpoint(graph, saved)

    async def read_latest(self, caller: TaskAddress | None) -> SavedCheckpoint | None:
        return await anext(self.checkpointer.list_checkpoints(self.thread_id, caller), None)

    async def find_interrupts(self, graph: AnyGraph, saved: SavedCheckpoint) -> list[PendingInterrupt]:
        reject_unknown_nodes(graph, (task.node for task in saved.checkpoint.tasks))
        pending: list[PendingInterrupt] = []
        for index, task in enumerate(saved.checkpoint.tasks):
            address = TaskAddress(checkpoint_id=saved.checkpoint.id, index=index)
            state = saved.task_states.get(index)
            match state:
                case InterruptedTask():
                    interrupt_id = InterruptId(task=address, index=len(state.answers))
                    interrupt = Interrupt(interrupt_id=interrupt_id, node=task.node, payload=state.payload)
                    pending.append(PendingInterrupt(interrupt, state.answers))
                case CompletedTask():
                    pass
                case None:
                    pending.extend(await self.find_child_interrupts(graph, task.node, address))
                case _:
                    assert_never(state)
        return pending

    async def find_child_interrupts(self, graph: AnyGraph, node: str, address: TaskAddress) -> list[PendingInterrupt]:
        action = graph.nodes[node].action
        match action:
            case CompiledGraph():
                child_saved = await self.read_latest(address)
                return [] if child_saved is None else await self.find_interrupts(action, child_saved)
            case CallPlan():
                return []
            case _:
                assert_never(action)


@dataclass(frozen=True)
class StateSnapshot[StateT]:
    checkpoint: Checkpoint
    values: StateT
    next: tuple[str, ...]
    interrupts: tuple[Interrupt, ...]


@dataclass(frozen=True, kw_only=True)
class Thread[StateT, InputT, OutputT]:
    graph: CompiledGraph[StateT, InputT, OutputT]
    checkpointer: Checkpointer
    thread_id: str

    @cached_property
    def journal(self) -> ThreadJournal:
        return ThreadJournal(self.checkpointer, self.thread_id)

    async def read_root(self, checkpoint_id: UUID | None) -> SavedCheckpoint | None:
        if checkpoint_id is None:
            return await self.journal.read_latest(None)
        saved = await self.checkpointer.read_checkpoint(self.thread_id, checkpoint_id)
        if saved is not None and saved.checkpoint.caller is not None:
            raise ValueError(f"checkpoint {checkpoint_id} belongs to a child graph, so it has no state of this graph")
        return saved

    async def require_root(self, checkpoint_id: UUID | None) -> SavedCheckpoint:
        saved = await self.read_root(checkpoint_id)
        if saved is None:
            missing = "checkpoint" if checkpoint_id is None else f"checkpoint {checkpoint_id}"
            raise LookupError(f"thread {self.thread_id!r} has no {missing}")
        return saved

    async def ainvoke(self, input: InputT, *, step_limit: int = 25) -> RunResult[OutputT]:
        latest = await self.journal.load_latest(self.graph, None)
        given: Mapping[str, object] = {} if latest is None else latest.values
        position = self.graph.start(given, extract_writes(input), caller=None, parent=latest)
        await self.journal.save(self.graph, position)
        return await self.graph.run(position, step_limit=step_limit, journal=self.journal, answers={})

    async def aresume(self, *answers: Answer, step_limit: int = 25) -> RunResult[OutputT]:
        saved = await self.require_root(None)
        position = decode_checkpoint(self.graph, saved)
        pending = {item.interrupt.interrupt_id: item for item in await self.journal.find_interrupts(self.graph, saved)}
        unknown = [answer.interrupt_id for answer in answers if answer.interrupt_id not in pending]
        if unknown:
            raise ValueError(f"{unknown!r} are not pending interrupts of thread {self.thread_id!r}")
        resumed = {
            answer.interrupt_id.task: (*pending[answer.interrupt_id].answers, to_json(answer.value).decode())
            for answer in answers
        }
        if len(resumed) != len(answers):
            raise ValueError("each pending interrupt takes one answer")
        return await self.graph.run(position, step_limit=step_limit, journal=self.journal, answers=resumed)

    async def areplay(self, checkpoint_id: UUID, *, step_limit: int = 25) -> RunResult[OutputT]:
        source = decode_checkpoint(self.graph, await self.require_root(checkpoint_id))
        position = source.follow(CheckpointSource.fork, source.values, source.tasks)
        await self.journal.save(self.graph, position)
        return await self.graph.run(position, step_limit=step_limit, journal=self.journal, answers={})

    async def aupdate_state(
        self, update: object, *, as_node: str | None = None, checkpoint_id: UUID | None = None
    ) -> UUID:
        source = decode_checkpoint(self.graph, await self.require_root(checkpoint_id))
        updates: Sequence[Mapping[str, object]]
        match as_node:
            case None:
                updates = [extract_writes(update)]
                tasks = source.tasks
            case str():
                outputs = close_step(self.graph, source, as_node, update)
                updates = [output.writes for output in outputs]
                tasks = plan_tasks(outputs)
            case _:
                assert_never(as_node)
        values = apply_writes(self.graph.channels, source.values, updates)
        position = source.follow(CheckpointSource.update, values, tasks)
        await self.journal.save(self.graph, position)
        return position.checkpoint_id

    async def snapshot(self, saved: SavedCheckpoint) -> StateSnapshot[StateT]:
        position = decode_checkpoint(self.graph, saved)
        completed = {index for index, state in saved.task_states.items() if state.kind is TaskStateKind.completed}
        interrupts = await self.journal.find_interrupts(self.graph, saved)
        return StateSnapshot(
            checkpoint=saved.checkpoint,
            values=self.graph.build_state(position.values),
            next=tuple(task.node for index, task in enumerate(position.tasks) if index not in completed),
            interrupts=tuple(item.interrupt for item in interrupts),
        )

    async def aget_state(self, checkpoint_id: UUID | None = None) -> StateSnapshot[StateT] | None:
        saved = await self.read_root(checkpoint_id)
        return None if saved is None else await self.snapshot(saved)

    async def ahistory(self) -> list[StateSnapshot[StateT]]:
        return [await self.snapshot(saved) async for saved in self.checkpointer.list_checkpoints(self.thread_id, None)]

    async def adelete(self) -> None:
        await self.checkpointer.delete_thread(self.thread_id)

    def invoke(self, input: InputT, *, step_limit: int = 25) -> RunResult[OutputT]:
        self.graph.reject_async_nodes("ainvoke")
        return run_coroutine(lambda: self.ainvoke(input, step_limit=step_limit))

    def resume(self, *answers: Answer, step_limit: int = 25) -> RunResult[OutputT]:
        self.graph.reject_async_nodes("aresume")
        return run_coroutine(lambda: self.aresume(*answers, step_limit=step_limit))

    def replay(self, checkpoint_id: UUID, *, step_limit: int = 25) -> RunResult[OutputT]:
        self.graph.reject_async_nodes("areplay")
        return run_coroutine(lambda: self.areplay(checkpoint_id, step_limit=step_limit))

    def update_state(self, update: object, *, as_node: str | None = None, checkpoint_id: UUID | None = None) -> UUID:
        return run_coroutine(lambda: self.aupdate_state(update, as_node=as_node, checkpoint_id=checkpoint_id))

    def get_state(self, checkpoint_id: UUID | None = None) -> StateSnapshot[StateT] | None:
        return run_coroutine(lambda: self.aget_state(checkpoint_id))

    def history(self) -> list[StateSnapshot[StateT]]:
        return run_coroutine(self.ahistory)

    def delete(self) -> None:
        run_coroutine(self.adelete)
