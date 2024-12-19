# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
from collections import namedtuple
from enum import Flag, auto
from functools import partial


class BaseEnumOptions(Flag):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    @property
    def cls(self):
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)


class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""

    pass


MethodMap = partial(namedtuple("MethodMap", ["value", "cls"]), auto())