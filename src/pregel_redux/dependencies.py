import inspect
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, MutableMapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, get_args, get_origin


class Scope(Enum):
    node = "node"
    run = "run"


@dataclass(frozen=True)
class Depends:
    dependency: Callable[..., object]
    scope: Scope = field(default=Scope.node, kw_only=True)


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


def request_from_parameter(function: Callable[..., object], parameter: inspect.Parameter) -> Depends:
    if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
        raise TypeError(f"parameter {parameter.name!r} of {function!r} cannot receive a dependency by keyword")
    metadata = get_args(parameter.annotation)[1:] if get_origin(parameter.annotation) is Annotated else ()
    markers = [item for item in metadata if isinstance(item, Depends)]
    match markers:
        case [Depends() as request]:
            return request
        case []:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} is not annotated with Depends")
        case _:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} has more than one Depends annotation")


def dependency_parameters(function: Callable[..., object], *, state_parameters: int) -> dict[str, Depends]:
    parameters = list(inspect.signature(function, eval_str=True).parameters.values())
    if len(parameters) < state_parameters:
        raise TypeError(f"{function!r} must accept the state as its first parameter")
    return {parameter.name: request_from_parameter(function, parameter) for parameter in parameters[state_parameters:]}


def value_opener(function: Callable[..., object]) -> Callable[..., AbstractAsyncContextManager[object]]:
    if inspect.isasyncgenfunction(function):
        return asynccontextmanager(function)
    if inspect.isgeneratorfunction(function):
        open_in_context = contextmanager(function)

        @asynccontextmanager
        async def open_generator_value(**arguments: object) -> AsyncGenerator[object]:
            with open_in_context(**arguments) as value:
                yield value

        return open_generator_value

    @asynccontextmanager
    async def open_returned_value(**arguments: object) -> AsyncGenerator[object]:
        result = function(**arguments)
        yield await result if inspect.isawaitable(result) else result

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
    function = overrides.get(request.dependency, request.dependency)
    parameters = dependency_parameters(function, state_parameters=0)
    for name, child in parameters.items():
        if request.scope is Scope.run and child.scope is Scope.node:
            raise ValueError(
                f"run-scoped dependency {function!r} requests node-scoped dependency {child.dependency!r} "
                f"through parameter {name!r}"
            )
        resolve_provider(child, overrides, resolved, (*path, request))
    resolved[request] = ResolvedProvider(request, function, value_opener(function), parameters)
    return resolved[request]


def plan_call(
    function: Callable[..., object], overrides: Mapping[Callable[..., object], Callable[..., object]]
) -> CallPlan:
    parameters = dependency_parameters(function, state_parameters=1)
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


async def call_with_dependencies(
    plan: CallPlan, state: object, values: MutableMapping[Depends, object], exits: AsyncExitStack
) -> object:
    await open_providers(plan.node_providers, values, exits)
    result = plan.function(state, **{name: values[request] for name, request in plan.parameters.items()})
    return await result if inspect.isawaitable(result) else result
