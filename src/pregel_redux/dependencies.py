import inspect
from collections.abc import AsyncGenerator, Callable, Iterable, Mapping, MutableMapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, get_origin


class Scope(Enum):
    node = "node"
    run = "run"


@dataclass(frozen=True)
class Provider:
    function: Callable[..., Any]
    scope: Scope


type Dependency = Callable[..., Any] | Provider


def provider(*, scope: Scope) -> Callable[[Callable[..., Any]], Provider]:
    def decorate(function: Callable[..., Any]) -> Provider:
        return Provider(function, scope)

    return decorate


@dataclass(frozen=True)
class Depends:
    dependency: Dependency


@dataclass(frozen=True)
class ResolvedProvider:
    key: Dependency
    function: Callable[..., Any]
    scope: Scope
    open_value: Callable[..., AbstractAsyncContextManager[Any]]
    parameters: Mapping[str, Dependency]


@dataclass(frozen=True)
class CallPlan:
    function: Callable[..., Any]
    parameters: Mapping[str, Dependency]
    run_providers: tuple[ResolvedProvider, ...]
    node_providers: tuple[ResolvedProvider, ...]


def dependency_from_parameter(function: Callable[..., Any], parameter: inspect.Parameter) -> Dependency:
    if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
        raise TypeError(f"parameter {parameter.name!r} of {function!r} cannot receive a dependency by keyword")
    annotation = parameter.annotation
    markers = (
        [item for item in annotation.__metadata__ if isinstance(item, Depends)]
        if get_origin(annotation) is Annotated
        else []
    )
    match markers:
        case [Depends(dependency=dependency)]:
            return dependency
        case []:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} is not annotated with Depends")
        case _:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} has more than one Depends annotation")


def dependency_parameters(function: Callable[..., Any], *, state_parameters: int) -> dict[str, Dependency]:
    parameters = list(inspect.signature(function, eval_str=True).parameters.values())
    if len(parameters) < state_parameters:
        raise TypeError(f"{function!r} must accept the state as its first parameter")
    return {
        parameter.name: dependency_from_parameter(function, parameter) for parameter in parameters[state_parameters:]
    }


def value_opener(function: Callable[..., Any]) -> Callable[..., AbstractAsyncContextManager[Any]]:
    if inspect.isasyncgenfunction(function):
        return asynccontextmanager(function)
    if inspect.isgeneratorfunction(function):
        open_in_context = contextmanager(function)

        @asynccontextmanager
        async def open_generator_value(**arguments: Any) -> AsyncGenerator[Any]:
            with open_in_context(**arguments) as value:
                yield value

        return open_generator_value

    @asynccontextmanager
    async def open_returned_value(**arguments: Any) -> AsyncGenerator[Any]:
        result = function(**arguments)
        yield await result if inspect.isawaitable(result) else result

    return open_returned_value


def resolve_provider(
    key: Dependency,
    overrides: Mapping[Dependency, Dependency],
    resolved: dict[Dependency, ResolvedProvider],
    path: tuple[Dependency, ...],
) -> ResolvedProvider:
    if key in resolved:
        return resolved[key]
    if key in path:
        raise ValueError(f"dependency cycle through {key!r}")
    target = overrides.get(key, key)
    match target:
        case Provider():
            function, scope = target.function, target.scope
        case _:
            function, scope = target, Scope.node
    parameters = dependency_parameters(function, state_parameters=0)
    for name, dependency in parameters.items():
        child = resolve_provider(dependency, overrides, resolved, (*path, key))
        if scope is Scope.run and child.scope is Scope.node:
            raise ValueError(
                f"run-scoped provider {function!r} requests node-scoped provider {child.function!r} "
                f"through parameter {name!r}"
            )
    resolved[key] = ResolvedProvider(key, function, scope, value_opener(function), parameters)
    return resolved[key]


def plan_call(function: Callable[..., Any], overrides: Mapping[Dependency, Dependency]) -> CallPlan:
    parameters = dependency_parameters(function, state_parameters=1)
    resolved: dict[Dependency, ResolvedProvider] = {}
    for dependency in parameters.values():
        resolve_provider(dependency, overrides, resolved, ())
    return CallPlan(
        function,
        parameters,
        run_providers=tuple(item for item in resolved.values() if item.scope is Scope.run),
        node_providers=tuple(item for item in resolved.values() if item.scope is Scope.node),
    )


async def open_providers(
    providers: Iterable[ResolvedProvider], values: MutableMapping[Dependency, Any], exits: AsyncExitStack
) -> None:
    for item in providers:
        if item.key in values:
            continue
        arguments = {name: values[dependency] for name, dependency in item.parameters.items()}
        values[item.key] = await exits.enter_async_context(item.open_value(**arguments))


async def call_with_dependencies(
    plan: CallPlan, state: object, values: MutableMapping[Dependency, Any], exits: AsyncExitStack
) -> Any:
    await open_providers(plan.node_providers, values, exits)
    result = plan.function(state, **{name: values[dependency] for name, dependency in plan.parameters.items()})
    return await result if inspect.isawaitable(result) else result
