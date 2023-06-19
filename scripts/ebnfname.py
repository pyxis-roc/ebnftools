#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 University of Rochester
#
# SPDX-License-Identifier: MIT

from ebnftools.ebnfgrammar import EBNFAnnotatedGrammar
from ebnftools.ebnfast import *
import ebnftools.anno.names as ebnfnames

import argparse
_DEBUG = False
_USING_NAMESPACE = True

# unused, may be useful?
def get_rule_symbols(obj, symbols = None):
    if symbols is None:
        symbols = []

    if isinstance(obj, Symbol):
        return obj.value
    elif isinstance(obj, String):
        return []
    elif isinstance(obj, Alternation):
        alt = []
        alt.append(get_rule_symbols(obj.expr[0]))
        alt.append(get_rule_symbols(obj.expr[1]))
        return alt
    elif isinstance(obj, Subtraction):
        symbols = [get_rule_symbols(obj.expr[0])]
        symbols.append(get_rule_symbols(obj.expr[1]))
        return symbols
    elif isinstance(obj, CharClass):
        return []
    elif isinstance(obj, Optional):
        return [get_rule_symbols(obj.expr)]
    elif isinstance(obj, Concat):
        return [get_rule_symbols(obj.expr)]
    elif isinstance(obj, Sequence):
        symbols = [get_rule_symbols(obj.expr[0])]
        symbols.extend([get_rule_symbols(obj.expr[1])])
        return symbols
    else:
        raise NotImplementedError(f"Don't support {obj}")

def get_default_name(construct, oldname, already_assigned, paths):
    if oldname is not None:
        return oldname

    assert hasattr(construct, '_treepos')

    if _USING_NAMESPACE:
        path = construct._treepos[1]
    else:
        path = construct._treepos

    # try to find parent who has a name
    for r in range(len(path), 0, -1):
        pp = path[:r]
        key = (construct._treepos[0], pp) if _USING_NAMESPACE else pp

        if key in paths:
            if hasattr(paths[key], '_name'):
                path_str = paths[key]._name + "_" + '_'.join([str(s) for s in path[r:]])
                break
    else:
        # nope, just use the path
        path_str = '_'.join([str(s) for s in path])

    if isinstance(construct, Symbol):
        name = construct.value
    elif isinstance(construct, String):
        name = f"str_{path_str}"
    elif isinstance(construct, Alternation):
        name = f"alt_{path_str}"
    elif isinstance(construct, Sequence):
        name = f"seq_{path_str}"
    elif isinstance(construct, Concat):
        name = f"mult_{path_str}"
    elif isinstance(construct, Optional):
        name = f"opt_{path_str}"
    elif isinstance(construct, CharClass):
        name = f"cc_{path_str}"
    elif isinstance(construct, Subtraction):
        name = f"sub_{path_str}"
    else:
        raise NotImplementedError(f"Don't know how to assign name to {construct}")

    if name in already_assigned:
        suffix = 1
        while f"{name}_{suffix}" in already_assigned:
            suffix += 1

        name = f"{name}_{suffix}"

    assert name not in already_assigned, f"{name}, {already_assigned}"

    already_assigned.add(name)

    return name


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Generate name annotations for EBNF rules")
    p.add_argument("ebnffile", help="EBNF specification")
    p.add_argument("rule", help="Rule to name (all for all)")
    p.add_argument("--ast", action="store_true", help="Show AST")
    p.add_argument("--debug", action="store_true", help="Enable debug output")

    args = p.parse_args()
    _DEBUG = args.debug

    with open(args.ebnffile, "r") as f:
        gr = f.read()

    pgr = EBNFAnnotatedGrammar()
    pgr.parse(gr)

    for r in pgr.rules:
        if r.lhs.value == args.rule or args.rule == 'all':
            print(r)
            if args.ast: print(repr(r))
            tp = pgr.compute_treepos(r, use_ns = _USING_NAMESPACE)
            aa = set()
            pgr.name_objects(tp.values(), get_default_name, aa, tp)

            if args.debug:
                for k, v in tp.items():
                    print(k, v, v._name)

            x = ebnfnames.NamesAnno(r.lhs.value, dict([(k, v._name) for k, v in tp.items()]))
            a = x.to_anno()
            print("@" + str(a))
