#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020,2021,2023 University of Rochester
#
# SPDX-License-Identifier: MIT

from ..ebnfanno import SExprList, Anno, Symbol, Number
from collections import OrderedDict

class NamesAnno(object):
    def __init__(self, rule_name, tree_pos_to_names):
        self.rule_name = rule_name

        for x in tree_pos_to_names:
            if isinstance(x[0], str):
                assert x[0] == self.rule_name, f"Namespace {x[0]} must match rule name {self.rule_name}"
                # strip namespaces
                self.tp2n = dict([(k[1], v) for k, v in tree_pos_to_names.items()])

            break

        if not hasattr(self, 'tp2n'):
            self.tp2n = tree_pos_to_names

    def name_objects(self, obj, oldname):
        assert hasattr(obj, '_treepos'), f"{obj} does not have a treepos"
        if obj._treepos in self.tp2n:
            return self.tp2n[obj._treepos]

        return oldname

    def to_anno(self, use_prefixes = True, use_sequences = True):
        def __prefix(path):
            for r in range(len(path)-1, 0, -1):
                prefix = path[:r]
                if prefix in self.tp2n:
                    return (Symbol(self.tp2n[prefix]), *[Number(x) for x in path[r:]])

            return [Number(x) for x in path]

        pos = sorted(self.tp2n.keys())

        out = [Symbol(self.rule_name)]
        prev_prefix = None
        sequence = False

        for p in pos:
            sequence = False
            if prev_prefix and len(p) == len(prev_prefix):
                if p[-1] == prev_prefix[-1]+1 and p[:-1] == prev_prefix[:-1]:
                    # they share the same prefix
                    sequence = True

            prev_prefix = p

            o = self.tp2n[p]

            pfx = __prefix(p) if use_prefixes else p
            if sequence and use_sequences:
                out[-1].value = (*out[-1].value, Symbol(o))
            else:
                out.append(SExprList(SExprList(*pfx), Symbol(o)))

        # TODO: order names in BFS order

        return Anno('NAMEPOS', out)

    @staticmethod
    def from_anno(anno):
        def decode_path_prefix(pp, name2pos):
            numtype = Number

            if len(pp.value) == 2:
                # name start
                if isinstance(pp.value[0], Symbol) and isinstance(pp.value[1], numtype):
                    if pp.value[0].value not in name2pos:
                        raise ValueError(f"Name '{pp.value[0]}' used in path prefix {pp}, but has not been defined. Names so far: {list(name2pos.keys())}")
                    return (*name2pos[pp.value[0].value], pp.value[1].value)
                elif isinstance(pp.value[0], numtype) and isinstance(pp.value[1], numtype):
                    return (pp.value[0].value, pp.value[1].value)
                else:
                    raise ValueError(f"Incorrect syntax for 2-length path prefix: {pp}")
            else:
                if not all(isinstance(x, numtype) for x in pp.value):
                    raise ValueError(f"Invalid syntax for path prefix (must be all numbers): {pp}")

                return tuple([x.value for x in pp.value])

        a = anno
        if len(a.value) < 3:
            raise ValueError(f"NAMEPOS annotations require rule name and position to name mappings: {a}")

        assert isinstance(anno, SExprList) and anno.value[0].value == 'NAMEPOS', f"{a} not a valid name-by-position annotation"

        if not isinstance(a.value[1], Symbol):
            raise ValueError(f"Expected Symbol: {a.value[1]}")

        rule_name = a.value[1].value
        pos2name= dict() #OrderedDict()
        name2pos = dict() #OrderedDict()

        for a in anno.value[2:]:
            if not isinstance(a, SExprList):
                raise ValueError(f"Expecting S-expression, found {a}")

            if len(a.value) < 2:
                # must be ((path_prefix) name1 name2 ...)
                raise ValueError(f"Invalid syntax for position to name mapping: {a}")

            pp = decode_path_prefix(a.value[0], name2pos)

            for r in range(len(a.value)-1):
                sym = a.value[r+1]
                if not isinstance(sym, Symbol):
                    raise ValueError(f"Name must be a symbol {sym} (in '{a}')")

                if sym.value in name2pos:
                    raise ValueError(f"WARNING: Name '{sym}' duplicated in {a}, this is okay if done at the same prefix level, but not checked yet!")

                npp = (*pp[:-1], pp[-1]+r)
                if npp in pos2name:
                    raise ValueError(f"Path prefix {npp} (specified as '{a}') is duplicated, {pos2name}")

                pos2name[npp] = sym.value
                name2pos[sym.value] = npp

        return NamesAnno(rule_name, pos2name)
