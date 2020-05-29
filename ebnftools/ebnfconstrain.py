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

try:
    from .ebnfast import *
    from .ebnfgen import *
except ImportError:
    from ebnfast import *
    from ebnfgen import *

import itertools

# expr ::= implication
#
# implication ::= xjunction '=>' expr

# xjunction ::=  term | term 'and' xjunction | term 'or' xjunction | '(' expr ')' | 'not' xjunction
#
# term ::= symbol '=' symbol | symbol '=' string

# based on smt2ast parser
class SExprList(object):
    def __init__(self, *args):
        self.value = args

    def __str__(self):
        return f"({' '.join([str(s) for s in self.value])})"

class ConstraintParser(object):
    def tokenize(self, data):
        # based on the example in the re docs
        tokens = [('STRING1', r'"[^"]*"'),
                  ('STRING2', r"'[^']*'"),
                  ('LPAREN', r'\('),
                  ('RPAREN', r'\)'),
                  ('SYMBOL', r'[A-Za-z_][A-Za-z0-9_]*'),
                  ('WHITESPACE', r'\s+'),
                  ('MISMATCH', r'.'),]

        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in tokens)

        for m in re.finditer(tok_regex, data, flags=re.M):
            token = m.lastgroup
            match = m.group()

            if token == 'MISMATCH':
                raise ValueError(f"Mismatch {match} (when parsing {data})")
            elif token == 'WHITESPACE':
                pass
            else:
                yield (token, match)

    def parse(self, llstr, token_stream = None):
        if token_stream is None:
            token_stream = self.tokenize(llstr)

        out = []
        try:
            while True:
                tkn, match = next(token_stream)
                if tkn == "RPAREN":
                    x = SExprList(*out)
                    if len(x.value):
                        sym = x.value[0].value
                        if sym  == "eq":
                            assert len(x.value) == 3, f"eq needs two arguments: {x}"
                            return Equals(x.value[1], x.value[2])
                        elif sym == "not":
                            assert len(x.value) == 2, f"not needs one argument: {x}"
                            return Not(x.value[1])
                        elif sym == "imp":
                            assert len(x.value) == 3, f"eq needs two arguments: {x}"
                            return Imp(x.value[1], x.value[2])
                        elif sym == "and":
                            assert len(x.value) >= 3, f"and needs at least two arguments: {x}"
                            return And(x.value[1:])
                        elif sym == "or":
                            assert len(x.value) >= 3, f"or needs at least two arguments: {x}"
                            return Or(x.value[1:])
                        else:
                            raise NotImplementedError(f"Unknown symbol {x.value[0]}, in {x}")
                    else:
                        raise ValueError(f"Empty list not supported")
                elif tkn == "LPAREN":
                    out.append(self.parse(llstr, token_stream))
                elif tkn == "SYMBOL":
                    out.append(Symbol(match))
                elif tkn == "STRING1" or tkn == "STRING2":
                    out.append(String(match[1:-1]))
                else:
                    raise NotImplementedError(f"Unknown token {tkn} '{match}'")
        except StopIteration:
            #raise ValueError("Ran out of input when parsing SExpr")
            pass

        if len(out) == 1:
            return out[0]
        else:
            raise ValueError(f"Parse resulted in multiple items! {out}")

def parse_constraint(line):
    cmd, r, cons = line.split(':', 3)

    cp = ConstraintParser()
    cons = cp.parse(cons)
    #TODO: full-fledged parser. For now, support only =
    #cons = cons.split("=")
    #cons = Equals(Symbol(cons[0].strip()), Symbol(cons[1].strip()))

    return ([rr.strip() for rr in r.split(",")], cons, cmd == "CONSTRAINT_DEBUG")

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
    debug_constraints = set()
    gr = []
    for l in src:
        if l.startswith("CONSTRAINT:") or l.startswith("CONSTRAINT_DEBUG:"):
            r, c, dbg = parse_constraint(l)
            for rr in r:
                if rr not in constraints:
                    constraints[rr] = []

                if dbg: debug_constraints.add(rr)
                constraints[rr].append(c)
        else:
            gr.append(l)

    ca = []
    for r in constraints:
        ca.append(Constraint(r, constraints[r], debug = r in debug_constraints))

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
    def __init__(self, rule, constraints, debug = False):
        self.rule = rule
        self.constraints = constraints
        self.debug = debug

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

        if self.c.debug:
            print("variables", vo)
            print("domains", self.domains)

        #vi = []
        #for ccv in self.cv:
        #    vi.append([vo.index(v) for v in ccv])

        #TODO: do some sort of domain reduction

        for assignment in itertools.product(*dom):
            if self.c.debug: print("assignment", assignment)
            asgn = dict(zip(vo, assignment))
            for ci, c in enumerate(self.c.constraints):
                if not c.check(asgn):
                    if self.c.debug: print("unsat", asgn, c)
                    break
            else:
                if self.c.debug: print("sat", asgn)
                yield asgn
                #print(rewrite_syms(self.c.grammar[self.c.rule], asgn))

class Not(object):
    def __init__(self, x):
        self.x = x

    def children(self):
        return [self.x]

    def check(self, assignment):
        return not self.x.check(assignment)

    def __str__(self):
        return f"!({self.x})"

class Imp(object):
    def __init__(self, antecedent, consequent):
        self.ant = antecedent
        self.con = consequent

    def children(self):
        return [self.ant, self.con]

    def check(self, assignment):
        return (not self.ant.check(assignment)) or self.con.check(assignment)

    def __str__(self):
        return f"({self.ant} => {self.con})"

class And(object):
    def __init__(self, terms):
        self.terms = terms

    def children(self):
        return self.terms

    def check(self, assignment):
        return all((t.check(assignment) for t in self.terms))

    def __str__(self):
        return " and ".join([f"({t})" for t in self.terms])

class Or(object):
    def __init__(self, terms):
        self.terms = terms

    def children(self):
        return self.terms

    def check(self, assignment):
        return any((t.check(assignment) for t in self.terms))

    def __str__(self):
        return " or ".join([f"({t})" for t in self.terms])

# only really works for primitive object comparisons
class Equals(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def children(self):
        return [self.lhs, self.rhs]

    def check(self, assignment):
        # TODO: generate the check to eliminate this if
        if isinstance(self.rhs, String):
            return assignment[self.lhs.value] == self.rhs.value
        else:
            return assignment[self.lhs.value] == assignment[self.rhs.value]

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"

def get_symbols(root, out = None):
    if out is None:
        out = []

    if isinstance(root, Symbol):
        out.append(root)
    elif isinstance(root, String):
        pass
    else:
        for x in root.children():
            get_symbols(x, out)

    return out

def rewrite_syms(root, varassign):
    def make_sequence(x):
        if len(x) == 2:
            return Sequence(String(x[0]), String(x[1]))
        elif len(x) == 1:
            return String(x[0])
        elif len(x) == 0:
            return String('')

        return Sequence(make_sequence(x[1:]), String(x[0]))

    if isinstance(root, Symbol):
        if root.value in varassign:
            rep = varassign[root.value]
            if isinstance(rep, str):
                return String(rep)
            else:
                return make_sequence(rep)
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
            print(f"WARNING: Constraints were not satisfied for {c.rule}, replacing with empty string")
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

def test_opt():
    gr = """
sep ::= '.'
ftz_clause ::= (sep 'ftz')?
type1 ::= 'f16' | 'f32' | 'f64'
x_opcode ::= 'somecode' ftz_clause type1

CONSTRAINT: x_opcode: (imp (not (eq type1 'f32')) (eq ftz_clause ''))
"""

    gr = gr.split('\n')

    constraints, gr, cgr = parse_constrained_grammar(gr, True, True)

    print(constraints)
    print('\n'.join([str(s) for s in gr]))
    print('\n'.join([str(s) for s in cgr]))

