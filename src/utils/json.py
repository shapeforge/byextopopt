from __future__ import absolute_import
import json
from enum import Enum


def get_enum_hook(global_vars):
    """
    Read a json entry as an Enum object.
    """

    def as_enum(d):
        if "__enum__" in d:
            name, member = d["__enum__"].split(".")
            return getattr(global_vars[name], member)
        else:
            return d

    return as_enum


class EnumEncoder(json.JSONEncoder):
    """
    Write an Enum object as a json dict.
    """

    def default(self, obj):
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


class SimpleEnumEncoder(json.JSONEncoder):
    """
    Write an Enum object as a json string.
    """

    def default(self, obj):
        if isinstance(obj, Enum):
            return str(obj)
        return json.JSONEncoder.default(self, obj)
