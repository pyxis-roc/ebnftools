#!/usr/bin/env python3
#
# ebnfconstrain.py
#
# Given a EBNF grammar and syntactic constraints that specify a valid
# subset of that grammar, generate a EBNF grammar that encodes those
# constraints.
#

# Restrictions on input grammars are indicated by expressions involving non-terminals.
#
# Non-terminals must have different names since there is no way to distinguish different instances
# of the same non-terminal in a EBNF rule.
#
# Supported constraints:
#   nonterminal1 = nonterminal2 # constraint values of both nonterminals to be equal
#   not constraint              # negate constraint

from .ebnfast import *
from .ebnfgen import *
import itertools

def parse_constraint(line):
    _, r, cons = line.split(':', 3)

    #TODO: full-fledged parser. For now, support only =
    cons = cons.split("=")
    cons = Equals(Symbol(cons[0].strip()), Symbol(cons[1].strip()))

    return (r.strip(), cons)

def parse_constrained_grammar(src, parse_grammar = True, apply_constraints = False):
    """A constrained grammar is a standard grammar that EBNFParse can
       parse plus embedded constraints, one per line, as:

       CONSTRAINT: rule: constraint in text.

       EBNFParse cannot parse these constrained grammars, and this
       function separates the two and returns parsed constraints and,
       optionally, the parsed grammar.

       If apply_constraints is True, then constrained grammar is also
       returned.
    """

    constraints = {}
    gr = []
    for l in src:
        if l.startswith("CONSTRAINT:"):
            r, c = parse_constraint(l)
            if r not in constraints:
                constraints[r] = []
            constraints[r].append(c)
        else:
            gr.append(l)

    ca = []
    for r in constraints:
        ca.append(Constraint(r, constraints[r]))

    cgr = None
    constraints = ca
    if apply_constraints or parse_grammar:
        p = EBNFParser()
        gr = '\n'.join(gr)
        gr = p.parse(gr)

        if apply_constraints:
            cgr = apply_grammar_constraints(constraints, gr)

    return constraints, gr, cgr

def apply_grammar_constraints(constraints, grammar):
    grd = dict([(rule.lhs.value, rule.rhs) for rule in grammar])
    apply_constraints(grd, constraints)

    out = []
    for r in grammar:
        out.append(Rule(r.lhs, grd[r.lhs.value]))

    return out

class Constraint(object):
    def __init__(self, rule, constraints):
        self.rule = rule
        self.constraints = constraints

    def __str__(self):
        return f"{self.rule}: {';'.join([str(s) for s in self.constraints])}"

class EnumerationSolver(object):
    def __init__(self, grammar, constraint):
        self.grammar = grammar
        self.c = constraint
        self.cv, self.variables = self._get_variables()
        self.domains = self._get_domains()

    def _get_variables(self):
        variables = set()
        cv = []
        for i, c in enumerate(self.c.constraints):
            v = set([x.value for x in get_symbols(c)])
            cv.append(v)
            variables = variables.union(v)

        return cv, variables

    def _get_domains(self):
        domains = {}
        for v in self.variables:
            domains[v] = set(generate2(self.grammar, self.grammar[v]))

        return domains

    def solve(self):
        vo = list(self.variables)
        dom = list(self.domains[v] for v in vo)

        vi = []
        for ccv in self.cv:
            vi.append([vo.index(v) for v in ccv])

        #print(vi)
        #TODO: do some sort of domain reduction

        for assignment in itertools.product(*dom):
            for ci, c in enumerate(self.c.constraints):
                if not c.check(vi[ci], assignment):
                    break
            else:
                asgn = dict(zip(vo, assignment))
                yield asgn
                #print(rewrite_syms(self.c.grammar[self.c.rule], asgn))

class Not(object):
    def __init__(self, x):
        self.x = x

    def children(self):
        return [self.x]

    def check(self, varindex, assignment):
        return not self.x.check(varindex, assignment)

    def __str__(self):
        return f"!({self.x})"

class Equals(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def children(self):
        return [self.lhs, self.rhs]

    def check(self, varindex, assignment):
        #TODO: generate the check

        return assignment[varindex[0]] == assignment[varindex[1]]

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"

class Implies(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def children(self):
        return [self.lhs, self.rhs]

    def __str__(self):
        return f"{self.lhs} => {self.rhs}"

def get_symbols(root, out = None):
    if out is None:
        out = []

    if isinstance(root, Symbol):
        out.append(root)
    else:
        for x in root.children():
            get_symbols(x, out)

    return out

def rewrite_syms(root, varassign):
    if isinstance(root, Symbol):
        if root.value in varassign:
            return String(varassign[root.value])
    elif isinstance(root, (Char, String, CharClass)):
        return root
    elif isinstance(root, Optional):
        return Optional(rewrite_syms(root.expr, varassign))
    elif isinstance(root, BinOp):
        ty = root.__class__
        return ty(rewrite_syms(root.expr[0], varassign),
                  rewrite_syms(root.expr[1], varassign))
    elif isinstance(root, Concat):
        return Concat(rewrite_syms(root.expr, varassign), root.minimum)

    return root

def apply_constraints(grammar, rule_constraints):
    for c in rule_constraints:
        es = EnumerationSolver(grammar, c)
        out = None
        for asgn in es.solve():
            rs = rewrite_syms(grammar[c.rule], asgn)
            if out is None:
                out = rs
            else:
                out = Alternation(out, rs)

        if out is None:
            print("WARNING: Constraints were not satisfied for {c.rule}, replacing with empty string")
            out = String('')

        grammar[c.rule] = out

def test_Constraints():
    gr = """
sep ::= '.'
shape ::= 'm8n8k32'
atype  ::= 's4' | 'u4'
btype  ::= 's4' | 'u4'

wmma_mma_opcode ::= 'wmma' sep 'mma' sep 'sync' sep 'aligned' sep 'row' sep 'col' sep shape sep s32 sep atype sep btype sep s32 (sep 'satfinite')?

"""

    p = EBNFParser()
    ast = p.parse(gr, as_dict=True)

    print(ast)
    c = Constraint("wmma_mma_opcode", [Equals(Symbol("atype"), Symbol("btype"))])
    apply_constraints(ast, [c])

    for a in ast:
        print(a, '::=', ast[a])

def test_parse_constrained_grammar():
    gr = """
sep ::= '.'
shape ::= 'm8n8k32'
atype  ::= 's4' | 'u4'
btype  ::= 's4' | 'u4'

wmma_mma_opcode ::= 'wmma' sep 'mma' sep 'sync' sep 'aligned' sep 'row' sep 'col' sep shape sep s32 sep atype sep btype sep s32 (sep 'satfinite')?

CONSTRAINT: wmma_mma_opcode: atype = btype
"""

    gr = gr.split('\n')

    constraints, gr, cgr = parse_constrained_grammar(gr, True, True)

    print(constraints)
    print('\n'.join([str(s) for s in gr]))
    print('\n'.join([str(s) for s in cgr]))

