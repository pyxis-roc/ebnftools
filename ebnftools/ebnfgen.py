#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020,2021,2023 University of Rochester
#
# SPDX-License-Identifier: MIT

try:
    from .ebnfast import *
except ImportError:
    from ebnfast import *

import itertools

def generate2(grammar, obj, _sym_stack = None):
    if _sym_stack is None:
        _sym_stack = set()

    if isinstance(obj, Symbol):
        if obj.value in _sym_stack: raise RecursionError(f"Generation of infinite productions by recursive use of symbol {obj.value} is not supported")
        _sym_stack.add(obj.value)
        x = generate2(grammar, grammar[obj.value], _sym_stack)
        _sym_stack.remove(obj.value)
        return x
    elif isinstance(obj, String):
        return [obj.value]
    elif isinstance(obj, Alternation):
        alt1 = generate2(grammar, obj.expr[0], _sym_stack)
        alt2 = generate2(grammar, obj.expr[1], _sym_stack)
        return itertools.chain(alt1, alt2)
    elif isinstance(obj, Subtraction):
        e1 = generate2(grammar, obj.expr[0], _sym_stack)
        e2 = set(generate2(grammar, obj.expr[1], _sym_stack))
        return filter(lambda x: x not in e2, e1)
    elif isinstance(obj, CharClass) and not obj.invert:
        return obj.iter()
    elif isinstance(obj, Optional):
        return itertools.chain(generate2(grammar, obj.expr, _sym_stack), [''])
    elif isinstance(obj, Sequence):
        seq1 = generate2(grammar, obj.expr[0], _sym_stack)
        seq2 = generate2(grammar, obj.expr[1], _sym_stack)
        return itertools.product(seq1, seq2)
    elif isinstance(obj, Concat):
        raise RecursionError(f"Generation of infinite productions through use of <{obj}> is not supported")
    else:
        raise NotImplementedError(f"Don't support {obj} ({type(obj)})")

def isfinite(grammar, obj, _symbol_stack = None):
    if _symbol_stack is None:
        # this is not really a stack, but since we're only searching
        # for duplicates, this is acceptable. A list would potentially
        # slow down the search.
        _symbol_stack = set()

    if isinstance(obj, (String, CharClass)):
        return True
    elif isinstance(obj, Symbol):
        if obj.value in _symbol_stack: return False # recursion!

        _symbol_stack.add(obj.value)
        x = isfinite(grammar, grammar[obj.value], _symbol_stack)
        _symbol_stack.remove(obj.value)

        return x
    elif isinstance(obj, BinOp):
        return isfinite(grammar, obj.expr[0], _symbol_stack) and isfinite(grammar, obj.expr[1], _symbol_stack)
    elif isinstance(obj, Optional):
        return isfinite(grammar, obj.expr, _symbol_stack)
    elif isinstance(obj, Concat):
        return False
    else:
        raise NotImplementedError(f"Missing {obj} in isfinite")

def count(grammar, obj):
    if isinstance(obj, Symbol):
        return count(grammar, grammar[obj.value])
    elif isinstance(obj, String):
        return 1
    elif isinstance(obj, Alternation):
        alt1 = count(grammar, obj.expr[0])
        alt2 = count(grammar, obj.expr[1])
        return alt1 + alt2
    elif isinstance(obj, Subtraction):
        e1 = generate2(grammar, obj.expr[0])
        e2 = set(generate2(grammar, obj.expr[1]))
        return len(list(filter(lambda x: x not in e2, e1)))
    elif isinstance(obj, CharClass) and not obj.invert:
        return len(list(obj.iter()))
    elif isinstance(obj, Optional):
        return count(grammar, obj.expr) + 1
    elif isinstance(obj, Sequence):
        seq1 = count(grammar, obj.expr[0])
        seq2 = count(grammar, obj.expr[1])
        return seq1 * seq2
    else:
        raise NotImplementedError(f"Don't support {obj}")

# visit the generated tuples
def visitor_make_sequence(s):
    if isinstance(s, str):
        return String(s)
    elif isinstance(s, (String, Sequence)):
        return s
    elif isinstance(s, tuple):
        return Sequence(visitor_make_sequence(s[0]), visitor_make_sequence(s[1]))
    else:
        raise NotImplementedError(f"make_sequence: unknown {s}/{type(s)}")

# visit the output of generate2
def visit_gen(s, visitor, visitor_args = []):
    if isinstance(s, str):
        return visitor(s, *visitor_args)
    elif isinstance(s, tuple):
        return visitor(s, *visitor_args)
    else:
        raise ValueError(f"Support only tuples and strings {s}")

def flatten(s, out = None):
    if out is None:
        out = []

    if isinstance(s, str):
        out.append(s)
    elif isinstance(s, list):
        assert len(s) == 0, f"Support only empty lists: {s}"
    elif isinstance(s, tuple):
        out = flatten(s[0], out)
        out = flatten(s[1], out)
    else:
        raise ValueError(f"Support only tuples and strings {s}")

    return out

def prefix(s, length, out = None):
    if out is None:
        out = []

    if len(out) == length:
        return out

    if isinstance(s, str):
        out.append(s)
    elif isinstance(s, list):
        assert len(s) == 0, f"Support only empty lists: {s}"
    elif isinstance(s, tuple):
        out = prefix(s[0], length, out)
        out = prefix(s[1], length, out)
    else:
        raise ValueError(f"Support only tuples and strings {s}")

    return out

def test_gen():
    grammar = """
    type ::= 'u16' | 'u32' | 'u64' | 's16' | 's32' | 's64'
    add ::= 'add' '.' (('sat' '.' 's32') | type )
    mul ::= 'mul' '.' ('hi' | 'lo' | 'wide') '.' type
    dp2a ::= 'dp2a' '.' ('lo' | 'hi') '.' dpXa_types '.' dpXa_types
    dp4a ::= 'dp4a' '.' dpXa_types '.' dpXa_types
    dpXa_types ::= 'u32' | 's32'
    addc ::= 'addc' ('.' 'cc')? '.' ('u32' | 's32' | 'u64' | 's64')
    simple ::= 'add' '.' 'sat'

    set ::= 'set' '.' set_CmpOp ('.' 'ftz')? '.' set_dtype '.' set_stype
    set_dtype ::= 'u32' | 's32' | 'f32'
    set_stype ::= ((('b' | 'u' | 's') ('16' | '32' | '64')) | 'f32' | 'f64')
    set_CmpOp ::= 'eq' | 'ne'
"""

    p = EBNFParser()
    r = p.parse(grammar)

    out = {}
    for rule in r:
        out[rule.lhs.value] = rule.rhs

    if False:
        g = generate_graph(r[8], generate_dot, out)
        with open("/tmp/g.dot", "w") as f:
            for l in g:
                print(l, file=f)

    gen = generate2(out, out['dp2a'])
    for l in gen:
        print(''.join(flatten(l)))

def test_visit_gen():
    p = EBNFParser()
    r = p.parse("set_stype ::= ((('b' | 'u' | 's') ('16' | '32' | '64')) | 'f32' | 'f64')", as_dict=True)

    gen = generate2(r, r['set_stype'])
    for l in gen:
        print(visit_gen(l, visitor_make_sequence))

def test_isfinite():
    p = EBNFParser()
    r = p.parse("a_star_bnf ::= '' | 'a' a_star_bnf", as_dict=True)

    assert not isfinite(r, r['a_star_bnf'])

def test_infinite_gen():
    p = EBNFParser()
    r = p.parse("""a_star_bnf ::= '' | 'a' a_star_bnf

a_star ::= 'a'*

""", as_dict=True)

    #TODO: make sure these are thrown
    try:
        generate2(r, r['a_star'])
    except RecursionError:
        pass

    try:
        generate2(r, r['a_star_bnf'])
    except RecursionError:
        pass

