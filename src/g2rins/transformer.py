# (C) 2025 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

import lark
from lark.visitors import Discard

from .exception import UnsupportedBigSMILES


class G2RINSTransformer(lark.Transformer):
    def stochastic_generation(self, children):
        return children[1]

    def NUMBER(self, children):
        return float(children)

    def DIGIT(self, children):
        return str(children)

    def INT(self, children):
        return int(children)

    def WS_INLINE(self, children):
        return Discard

    def fragment_definition(self, children):
        raise UnsupportedBigSMILES("fragment_definition", children)

    def fragment_declaration(self, children):
        raise UnsupportedBigSMILES("fragment_declaration", children)

    def ladder_bond_connector(self, children):
        raise UnsupportedBigSMILES("ladder_bond_connector", children)

    def inner_non_covalent_connector(self, children):
        raise UnsupportedBigSMILES("inner_non_covalent_connector", children)

    def inner_ambi_covalent_connector(self, children):
        raise UnsupportedBigSMILES("inner_ambi_covalent_connector", children)

    def non_covalent_bond_connector(self, children):
        raise UnsupportedBigSMILES("non_covalent_bond_connector", children)


_GLOBAL_TRANSFORMER: None | G2RINSTransformer = None


def get_global_transformer():
    global _GLOBAL_TRANSFORMER
    if _GLOBAL_TRANSFORMER is None:
        import g2rins

        transformer = lark.ast_utils.create_transformer(ast_module=g2rins, transformer=G2RINSTransformer(visit_tokens=True))
        _GLOBAL_TRANSFORMER = transformer

    return _GLOBAL_TRANSFORMER
