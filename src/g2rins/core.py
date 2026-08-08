# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

from abc import ABC, ABCMeta, abstractmethod

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from lark import ast_utils

from .generative_graph import _PartialGraph
from .parser import get_global_parser
from .transformer import get_global_transformer
from .util import camel_to_snake


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        if obj is None:
            return self.f(owner)
        return self.f(obj)


class MetaSkipInitIfChildIsInstance(ABCMeta):
    def __call__(cls, children):
        # Special allocation function. For inherited objects.
        # If the inherited object is already fully created, we don't create a new object, but just use that child
        if len(children) == 1 and isinstance(children[0], cls):
            return children[0]

        return super().__call__(children)


class G2rinsBase(ABC, ast_utils.Ast, ast_utils.AsList, metaclass=MetaSkipInitIfChildIsInstance):
    @classmethod
    def make(cls, text: str) -> Self:
        tree = get_global_parser().parse(text, start=cls.token_name_snake_case)
        transformed_tree = get_global_transformer().transform(tree)
        return transformed_tree

    def __init__(self, children: list):
        super().__init__()

        if len(children) > 0 and children[0] is self:
            return

        self._children = children

    @classproperty
    def token_name(self) -> str:
        name = type(self).__name__
        if name in ("ABCMeta", "MetaSkipInitIfChildIsInstance"):
            name = self.__name__
        return name

    @classproperty
    def token_name_snake_case(self) -> str:
        return camel_to_snake(self.token_name)

    def __str__(self) -> str:
        return self.generate_string(True)

    def generate_string(self, extension: bool) -> str:
        raise NotImplementedError(f"Base class G2rinsBase does not implement generate_string. If you see this please report on github. {type(self)}")

    @property
    def generable(self) -> bool:
        raise NotImplementedError(f"Base class G2rinsBase does not implement generable. If you see this please report on github. {type(self)}")

    @property
    def residues(self) -> list:
        return []

    @property
    def bond_connectors(self) -> list:
        return []

    @property
    def molar_amount(self) -> float:
        return 1.0


class GenerationBase(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _generate_partial_graph(self) -> _PartialGraph:
        pass

    def get_graph_creator(self):
        # Fresh on every call: node ids are minted per creator, so hold ONE creator
        # to get a generative graph and an ensemble creator with matching node ids.
        from .generative_graph import GraphCreator

        return GraphCreator(self._generate_partial_graph(), str(self))
