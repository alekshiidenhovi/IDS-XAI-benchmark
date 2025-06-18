import typing as T
import click


def parse_int_list(
    ctx: click.Context, param: click.Option, value: T.Optional[str]
) -> T.Optional[T.List[int]]:
    if value is None:
        return None
    return [int(x.strip()) for x in value.split(",")]
