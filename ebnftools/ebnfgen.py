#!/usr/bin/env python3

from .ebnfast import *
import itertools

def generate2(grammar, obj):
    if isinstance(obj, Symbol):
        return generate2(grammar, grammar[obj.value])
    elif isinstance(obj, String):
        return [obj.value]
    elif isinstance(obj, Alternation):
        alt1 = generate2(grammar, obj.expr[0])
        alt2 = generate2(grammar, obj.expr[1])
        return itertools.chain(alt1, alt2)
    elif isinstance(obj, Subtraction):
        e1 = generate2(grammar, obj.expr[0])
        e2 = set(generate2(grammar, obj.expr[1]))
        return filter(lambda x: x not in e2, e1)
    elif isinstance(obj, CharClass) and not obj.invert:
        return obj.iter()
    elif isinstance(obj, Optional):
        return itertools.chain(generate2(grammar, obj.expr), [[]])
    elif isinstance(obj, Sequence):
        seq1 = generate2(grammar, obj.expr[0])
        seq2 = generate2(grammar, obj.expr[1])
        return itertools.product(seq1, seq2)
    else:
        raise NotImplementedError(f"Don't support {obj}")

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

