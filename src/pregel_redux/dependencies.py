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


class Scope(Enum):
    node = "node"
    run = "run"


@dataclass(frozen=True)
class Depends:
    dependency: Callable[..., object]
    scope: Scope = field(default=Scope.node, kw_only=True)


def read_state() -> object:
    """Marker that requests the state of the current node call."""
    raise RuntimeError("read_state supplies a value only as a node-scoped dependency of a graph")


type StateDep[StateT] = Annotated[StateT, Depends(read_state)]


def expand_alias(annotation: object) -> object:
    match annotation:
        case TypeAliasType():
            return expand_alias(annotation.__value__)
        case _:
            origin = get_origin(annotation)
            return expand_alias(origin) if isinstance(origin, TypeAliasType) else annotation


@dataclass(frozen=True)
class ResolvedProvider:
    request: Depends
    function: Callable[..., object]
    open_value: Callable[..., AbstractAsyncContextManager[object]]
    parameters: Mapping[str, Depends]


@dataclass(frozen=True)
class CallPlan:
    function: Callable[..., object]
    parameters: Mapping[str, Depends]
    run_providers: tuple[ResolvedProvider, ...]
    node_providers: tuple[ResolvedProvider, ...]


def read_request(function: Callable[..., object], parameter: inspect.Parameter) -> Depends:
    if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
        raise TypeError(f"parameter {parameter.name!r} of {function!r} cannot receive a dependency by keyword")
    annotation = expand_alias(parameter.annotation)
    metadata = get_args(annotation)[1:] if get_origin(annotation) is Annotated else ()
    markers = [item for item in metadata if isinstance(item, Depends)]
    match markers:
        case [Depends() as request]:
            return request
        case []:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} is not annotated with Depends")
        case _:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} has more than one Depends annotation")


def read_dependency_parameters(function: Callable[..., object], *, state_parameters: int) -> dict[str, Depends]:
    parameters = list(inspect.signature(function, eval_str=True).parameters.values())
    if len(parameters) < state_parameters:
        raise TypeError(f"{function!r} must accept the state as its first parameter")
    return {parameter.name: read_request(function, parameter) for parameter in parameters[state_parameters:]}


async def finish_despite_cancellation[ResultT](work: Awaitable[ResultT]) -> ResultT:
    future = asyncio.ensure_future(work)
    cancelled = False
    while not future.done():
        try:
            await asyncio.wait([future])
        except asyncio.CancelledError:
            cancelled = True
    result = future.result()
    task = asyncio.current_task()
    if cancelled and task is not None:
        task.cancel()
    return result


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
    resolved: dict[Depends, ResolvedProvider],
    path: tuple[Depends, ...],
) -> ResolvedProvider:
    if request in resolved:
        return resolved[request]
    if request in path:
        raise ValueError(f"dependency cycle through {request.dependency!r}")
    if request.dependency is read_state and request.scope is Scope.run:
        raise ValueError("the state changes at each step, so a run-scoped dependency cannot request it")
    function = overrides.get(request.dependency, request.dependency)
    parameters = read_dependency_parameters(function, state_parameters=0)
    for name, child in parameters.items():
        if request.scope is Scope.run and child.scope is Scope.node:
            raise ValueError(
                f"run-scoped dependency {function!r} requests node-scoped dependency {child.dependency!r} "
                f"through parameter {name!r}"
            )
        resolve_provider(child, overrides, resolved, (*path, request))
    resolved[request] = ResolvedProvider(request, function, build_value_opener(function), parameters)
    return resolved[request]


def plan_call(
    function: Callable[..., object], overrides: Mapping[Callable[..., object], Callable[..., object]]
) -> CallPlan:
    parameters = read_dependency_parameters(function, state_parameters=1)
    resolved: dict[Depends, ResolvedProvider] = {}
    for request in parameters.values():
        resolve_provider(request, overrides, resolved, ())
    return CallPlan(
        function,
        parameters,
        run_providers=tuple(item for item in resolved.values() if item.request.scope is Scope.run),
        node_providers=tuple(item for item in resolved.values() if item.request.scope is Scope.node),
    )


async def open_providers(
    providers: Iterable[ResolvedProvider], values: MutableMapping[Depends, object], exits: AsyncExitStack
) -> None:
    for item in providers:
        if item.request in values:
            continue
        arguments = {name: values[request] for name, request in item.parameters.items()}
        values[item.request] = await exits.enter_async_context(item.open_value(**arguments))


async def execute_plan(
    plan: CallPlan, state: object, values: MutableMapping[Depends, object], exits: AsyncExitStack
) -> object:
    values[Depends(read_state)] = state
    await open_providers(plan.node_providers, values, exits)
    return await call_without_blocking(
        plan.function, state, **{name: values[request] for name, request in plan.parameters.items()}
    )
