from typing import Any, Callable, Type, TypeVar
from dataclasses import asdict
from pathlib import Path
from enum import Enum
import json
import os

from dacite import Config, from_dict
import pandas as pd


T = TypeVar("T")

DEFAULT_CONFIG = Config(cast=[Enum, tuple])


def enum_dict_factory(data) -> dict:
    """
    Recursively checks for Enums and converts any that are found to their respective
    values. NOTE: Recursion operates only on objects of type Dict and List.
    """
    new_data = []
    for k, v in data:
        if isinstance(v, Enum):
            new_data.append((k, v.value))
        elif isinstance(v, dict):
            new_data.append((k, enum_dict_factory(list(v.items()))))
        elif isinstance(v, list):
            result = enum_dict_factory([(kk, vv) for kk, vv in enumerate(v)])
            new_data.append((k, [result[i] for i in range(len(v))]))
        else:
            new_data.append((k, v))
    return dict(new_data)


def ensure_path(file_path: str) -> str:
    """Creates all parent folders of a file path (if needed). Returns the file path."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_lines(file_path: str, *objs: T, to_string_fn: Callable[[T], str] = str):
    """Saves each object as a string to the file path, one object per line."""
    str_objs = [to_string_fn(o) + os.linesep for o in objs]
    with open(ensure_path(file_path), "w", encoding="utf-8") as f:
        f.writelines(str_objs)


def save_json(file_path: str, obj, **kwargs):
    """Saves the object to the file path. kwargs are passed to json.dump()."""
    with open(ensure_path(file_path), "w", encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)


def save_jsonl(file_path: str, *objs: Any, **kwargs):
    """
    Saves each object as a JSON string to the file path, one object per line.
    kwargs are passed to json.dumps().
    """
    save_lines(file_path, *objs, to_string_fn=lambda o: json.dumps(o, **kwargs))


def save_dataclass_json(
    file_path: str,
    obj: Any,
    dict_factory: Callable = enum_dict_factory,
    **kwargs,
):
    """
    Saves the dataclass object to the file path. kwargs are passed to json.dumps().
    dict_factory is passed to dataclasses.asdict().
    """
    with open(ensure_path(file_path), "w", encoding="utf-8") as f:
        json.dump(asdict(obj, dict_factory=dict_factory), f, **kwargs)


def save_dataclass_jsonl(
    file_path: str,
    *objs: Any,
    dict_factory: Callable = enum_dict_factory,
    **kwargs,
):
    """
    Saves each dataclass object as a JSON string to the file path, one object per line.
    kwargs are passed to json.dump(). dict_factory is passed to dataclasses.asdict().
    """

    def helper(o):
        return json.dumps(asdict(o, dict_factory=dict_factory), **kwargs)

    save_lines(file_path, *objs, to_string_fn=helper)


def load_json(file_path: str, **kwargs) -> Any:
    """Loads a JSON object from the file path. kwargs are passed to json.load()."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f, **kwargs)


def load_jsonl(file_path: str, **kwargs) -> list[Any]:
    """Loads JSON objects from the file path. kwargs are passed to json.loads()."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip(), **kwargs) for line in f.readlines()]


def load_dataclass_json(
    file_path: str,
    t: Type[T],
    dacite_config: Config = DEFAULT_CONFIG,
    **kwargs,
) -> T:
    """
    Loads a dataclass object of type 't' from the file path. kwargs are passed
    to json.load(). dacite_config is passed to dacite.from_dict().
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return from_dict(t, json.load(f, **kwargs), config=dacite_config)


def load_dataclass_jsonl(
    file_path: str,
    t: Type[T],
    dacite_config: Config = DEFAULT_CONFIG,
    **kwargs,
) -> list[T]:
    """
    Loads dataclass objects of type 't' from the file path. kwargs are passed
    to json.loads(). dacite_config is passed to dacite.from_dict().
    """

    def helper(s):
        return from_dict(t, json.loads(s, **kwargs), config=dacite_config)

    with open(file_path, "r", encoding="utf-8") as f:
        return [helper(line.strip()) for line in f.readlines()]


def load_records_csv(file_path: str, **kwargs) -> list[dict]:
    """
    Loads records (one per CSV row) from the file path. kwargs are passed to
    pandas.read_csv().
    """
    return pd.read_csv(file_path, **kwargs).to_dict(orient="records")
