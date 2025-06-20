import datetime
import typing as T
import uuid

import click


def parse_int_list(
    ctx: click.Context, param: click.Option, value: T.Optional[str]
) -> T.Optional[T.List[int]]:
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",")]


def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_experiment_name(max_depth: int, n_estimators: int) -> str:
    current_datetime = get_current_datetime()
    return f"{current_datetime}-training-time-benchmark-max_depth-{max_depth}-n_estimators-{n_estimators}"


def get_benchmark_id() -> str:
    return str(uuid.uuid4())
