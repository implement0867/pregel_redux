import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, MutableMapping
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass, field
from enum import Enum
from types import TracebackType
from typing import Annotated, TypeAliasType, get_args, get_origin

from typing_extensions import TypeForm

from pregel_redux.channels import Stored, substitute_type_arguments


class Scope(Enum):
    """Sets the lifetime of a dependency value.

    `node` opens a value for each node call and closes it after the step.
    `run` opens a value once for each run and closes it at the end of the run.
    """

    node = "node"
    run = "run"


@dataclass(frozen=True)
class Depends:
    """Requests the value that `dependency` provides for a node parameter.

    The graph calls `dependency` with the values of its own parameters.
    A generator function provides its first yielded value.
    The remaining code of a generator function runs at the end of `scope`.
    A synchronous function runs in a worker thread.

    Parameters
    ----------
    scope
        A run-scoped dependency requests run-scoped values only.
    """

    dependency: Callable[..., object]
    scope: Scope = field(default=Scope.node, kw_only=True)


class NodeValue(Enum):
    state = "state"
    received = "received"


type StateDep[StateT] = Annotated[StateT, NodeValue.state]
type Received[MessageT] = Annotated[MessageT, NodeValue.received]
type Supplied = NodeValue | type[object]
type Request = Depends | Supplied


def expand_alias[HintT](annotation: HintT) -> HintT:
    match annotation:
        case TypeAliasType():
            return expand_alias(annotation.__value__)
        case _:
            origin = get_origin(annotation)
            match origin:
                case TypeAliasType():
                    arguments: dict[object, TypeForm[object]] = dict(
                        zip(origin.__type_params__, get_args(annotation), strict=True)
                    )
                    return expand_alias(substitute_type_arguments(origin.__value__, arguments))
                case _:
                    return annotation


@dataclass(frozen=True)
class ResolvedProvider:
    request: Depends
    function: Callable[..., object]
    open_value: Callable[..., AbstractAsyncContextManager[object]]
    parameters: Mapping[str, Request]


@dataclass(frozen=True)
class CallPlan:
    function: Callable[..., object]
    parameters: Mapping[str, Request]
    run_providers: tuple[ResolvedProvider, ...]
    node_providers: tuple[ResolvedProvider, ...]
    received: TypeForm[object] | None


def read_request(
    function: Callable[..., object], parameter: inspect.Parameter, supplied: Mapping[Supplied, Scope]
) -> Request:
    if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
        raise TypeError(f"parameter {parameter.name!r} of {function!r} cannot receive a dependency by keyword")
    annotation = expand_alias(parameter.annotation)
    if isinstance(annotation, type):
        if annotation not in supplied:
            raise TypeError(
                f"parameter {parameter.name!r} of {function!r} is not annotated with Depends, "
                f"and this graph does not supply a value of type {annotation!r}"
            )
        return annotation
    metadata = get_args(annotation)[1:] if get_origin(annotation) is Annotated else ()
    markers = [item for item in metadata if isinstance(item, Depends | NodeValue)]
    match markers:
        case [Depends() | NodeValue() as request]:
            return request
        case []:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} is not annotated with Depends")
        case _:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} has more than one Depends annotation")


def read_dependency_parameters(
    function: Callable[..., object], supplied: Mapping[Supplied, Scope], *, state_parameters: int
) -> dict[str, Request]:
    parameters = list(inspect.signature(function, eval_str=True).parameters.values())
    if len(parameters) < state_parameters:
        raise TypeError(f"{function!r} must accept the state as its first parameter")
    return {parameter.name: read_request(function, parameter, supplied) for parameter in parameters[state_parameters:]}


def read_received_annotation(
    function: Callable[..., object], requests: Mapping[str, Request]
) -> TypeForm[object] | None:
    parameters = inspect.signature(function, eval_str=True).parameters
    annotations = [
        expand_alias(parameters[name].annotation) for name, request in requests.items() if request is NodeValue.received
    ]
    match annotations:
        case []:
            return None
        case [first, *others] if all(other == first for other in others):
            return first
        case _:
            raise TypeError(f"{function!r} receives its message as more than one type: {annotations!r}")


async def finish_despite_cancellation[ResultT](work: Awaitable[ResultT]) -> ResultT:
    future = asyncio.ensure_future(work)
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("finish_despite_cancellation works only inside a task")
    requests = 0
    while not future.done():
        try:
            await asyncio.wait([future])
        except asyncio.CancelledError:
            requests += 1
            task.uncancel()
    if not future.cancelled() and (error := future.exception()) is not None:
        context = error.__context__
        try:
            raise error
        finally:
            error.__context__ = context
    result = future.result()
    for _ in range(requests):
        task.cancel()
    return result


class ShieldedExitStack(AsyncExitStack[bool]):
    async def __aexit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        return await finish_despite_cancellation(super().__aexit__(error_type, error, traceback))


def close_at_exit(exits: AsyncExitStack, until: AsyncExitStack) -> None:
    async def close(
        error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        try:
            await exits.aclose()
        except BaseException as cleanup_error:
            last = cleanup_error
            chain = {id(last)}
            while last.__context__ is not None and id(last.__context__) not in chain:
                last = last.__context__
                chain.add(id(last))
            if error is not None and id(error) not in chain and last.__context__ is None:
                last.__context__ = error
            raise
        return False

    until.push_async_exit(close)


async def run_in_thread(function: Callable[..., object], *arguments: object, **named_arguments: object) -> object:
    await asyncio.sleep(0)  # deliver a pending cancellation
    return await finish_despite_cancellation(asyncio.to_thread(function, *arguments, **named_arguments))


@dataclass(frozen=True)
class ThreadedContext(AbstractAsyncContextManager[object]):
    manager: AbstractContextManager[object]

    async def __aenter__(self) -> object:
        return await run_in_thread(self.manager.__enter__)

    async def __aexit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return await finish_despite_cancellation(asyncio.to_thread(self.manager.__exit__, error_type, error, traceback))


async def call_without_blocking(
    function: Callable[..., object], *arguments: object, **named_arguments: object
) -> object:
    if inspect.iscoroutinefunction(function):
        result = function(*arguments, **named_arguments)
    else:
        result = await run_in_thread(function, *arguments, **named_arguments)
    return await result if inspect.isawaitable(result) else result


def build_value_opener(function: Callable[..., object]) -> Callable[..., AbstractAsyncContextManager[object]]:
    if inspect.isasyncgenfunction(function):
        return asynccontextmanager(function)
    if inspect.isgeneratorfunction(function):
        open_in_context = contextmanager(function)

        def open_generator_value(**arguments: object) -> ThreadedContext:
            return ThreadedContext(open_in_context(**arguments))

        return open_generator_value

    @asynccontextmanager
    async def open_returned_value(**arguments: object) -> AsyncGenerator[object]:
        yield await call_without_blocking(function, **arguments)

    return open_returned_value


def resolve_provider(
    request: Depends,
    overrides: Mapping[Callable[..., object], Callable[..., object]],
    supplied: Mapping[Supplied, Scope],
    resolved: dict[Depends, ResolvedProvider],
    path: tuple[Depends, ...],
) -> ResolvedProvider:
    if request in resolved:
        return resolved[request]
    if request in path:
        raise ValueError(f"dependency cycle through {request.dependency!r}")
    function = overrides.get(request.dependency, request.dependency)
    parameters = read_dependency_parameters(function, supplied, state_parameters=0)
    for name, child in parameters.items():
        match child:
            case Depends():
                child_scope, child_name = child.scope, repr(child.dependency)
                resolve_provider(child, overrides, supplied, resolved, (*path, request))
            case _:
                child_scope, child_name = supplied[child], f"the value {child!r} of a node call"
        if request.scope is Scope.run and child_scope is Scope.node:
            raise ValueError(
                f"run-scoped dependency {function!r} requests node-scoped dependency {child_name} "
                f"through parameter {name!r}"
            )
    resolved[request] = ResolvedProvider(request, function, build_value_opener(function), parameters)
    return resolved[request]


def plan_call(
    function: Callable[..., object],
    overrides: Mapping[Callable[..., object], Callable[..., object]],
    supplied: Mapping[Supplied, Scope],
) -> CallPlan:
    parameters = read_dependency_parameters(function, supplied, state_parameters=1)
    resolved: dict[Depends, ResolvedProvider] = {}
    for request in parameters.values():
        match request:
            case Depends():
                resolve_provider(request, overrides, supplied, resolved, ())
            case _:
                pass
    received = read_received_annotation(function, parameters)
    readers = [item.function for item in resolved.values() if NodeValue.received in item.parameters.values()]
    if received is None and readers:
        raise TypeError(
            f"the dependencies {readers!r} of {function!r} request the received message, "
            f"and {function!r} has no Received parameter that gives its type"
        )
    return CallPlan(
        function,
        parameters,
        run_providers=tuple(item for item in resolved.values() if item.request.scope is Scope.run),
        node_providers=tuple(item for item in resolved.values() if item.request.scope is Scope.node),
        received=received,
    )


async def open_providers(
    providers: Iterable[ResolvedProvider], values: MutableMapping[Request, object], exits: AsyncExitStack
) -> None:
    for item in providers:
        if item.request in values:
            continue
        arguments = {name: values[request] for name, request in item.parameters.items()}
        values[item.request] = await exits.enter_async_context(item.open_value(**arguments))


async def execute_plan(
    plan: CallPlan,
    state: object,
    received: Stored[object] | None,
    values: MutableMapping[Request, object],
    exits: AsyncExitStack,
) -> object:
    values[NodeValue.state] = state
    if received is not None:
        values[NodeValue.received] = received.value
    await open_providers(plan.node_providers, values, exits)
    return await call_without_blocking(
        plan.function, state, **{name: values[request] for name, request in plan.parameters.items()}
    )
