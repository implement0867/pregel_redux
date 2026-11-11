import random
from collections.abc import Callable
from dataclasses import dataclass


def retry_by_default(failure: Exception) -> bool:
    match failure:
        case ConnectionError():
            return True
        case (
            ValueError()
            | TypeError()
            | ArithmeticError()
            | ImportError()
            | LookupError()
            | NameError()
            | SyntaxError()
            | RuntimeError()
            | ReferenceError()
            | StopIteration()
            | StopAsyncIteration()
            | OSError()
        ):
            return False
        case _:
            return True


@dataclass(frozen=True, kw_only=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    jitter: bool = True
    retry_on: type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], bool] = retry_by_default

    def allows(self, failure: Exception, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        match self.retry_on:
            case type() | tuple() as types:
                return isinstance(failure, types)
            case decide:
                return decide(failure)

    def read_delay(self, attempt: int) -> float:
        delay = min(self.max_interval, self.initial_interval * self.backoff_factor ** (attempt - 1))
        return delay + random.uniform(0, 1) if self.jitter else delay
