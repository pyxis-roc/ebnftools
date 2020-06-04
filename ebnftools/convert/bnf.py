#!/usr/bin/env python3
#
# bnf.py
#
# Convert a EBNF AST to BNF (though it is still EBNF).
#

# essentially, this replaces
# symbols -> symbols
# char -> char
# CharClasses must be explicitly replaced by TOKENS
# sequences are unchanged (except paren sequences need to be symbolized?)
# non-top-level x | y are replaced with bnf_x_y ::= x | y
# subtraction x - y are replaced by the set x - y if x and y can be materialized
# x* with a rule: bnf_x ::= empty | x bnf_x
# x+ with a rule: bnf_x ::= x bnf_x
# x? with a rule: bnf_x ::= empty | x
#
# String Literals are replaced by TOKENS

try:
    from ..ebnfast import *
    from ..ebnfgen import isfinite as ebnf_isfinite, generate2, visit_gen, visitor_make_sequence
except ValueError:
    from ebnftools.ebnfast import *
    from ebnftools.ebnfgen import isfinite as ebnf_isfinite, generate2, visit_gen, visitor_make_sequence

class LiteralRewriter(EBNFTransformer):
    def visit_String(self, node):
        sk = repr(node.value)
        if sk in self._literals_to_tokens:
            return Symbol(self._literals_to_tokens[sk])

        return node

    def rewrite(self, rules, lit2tokens):
        self._literals_to_tokens = lit2tokens
        return self.visit_RuleList(rules)

class CharClassRewriter(EBNFTransformer):
    def visit_CharClass(self, node):
        sk = str(node)
        if sk in self._literals_to_tokens:
            return Symbol(self._literals_to_tokens[sk])

        return node

    def rewrite(self, rules, lit2tokens):
        self._literals_to_tokens = lit2tokens
        return self.visit_RuleList(rules)

class EBNF2BNF(EBNFTransformer):
    dedup_across_rules = False

    def _new_name(self, name):
        x = 1
        if not self.dedup_across_rules:
            tryname = name + "_" + self.rulename
        else:
            tryname = name

        while tryname in self.names:
            tryname = name + f"_{x}"
            x += 1
        name = tryname
        self.names.add(name)
        return name

    def _new_rule(self, rule):
        k = str(rule.rhs)
        if k not in self._dedup_rules:
            self.new_rules.append(rule)
            self._dedup_rules[k] = rule.lhs

        return self._dedup_rules[k]

    def visit_Rule(self, node):
        self.new_rules = []
        if not self.dedup_across_rules:
            self._dedup_rules = {}
        self.rulename = node.lhs.value
        node = super().visit_Rule(node)

        if isinstance(node, list):
            self.new_rules.extend(node)
        else:
            self.new_rules.append(node)

        #print("***", self.new_rules)
        return self.new_rules

    def visit_Sequence(self, node):
        self.stack.append('seq')
        node = super().visit_Sequence(node)
        self.stack.pop()
        return node

    def visit_Subtraction(self, node):
        if ebnf_isfinite(self.rules_as_dict, node.expr[0]) and ebnf_isfinite(self.rules_as_dict,
                                                                             node.expr[1]):
            out = None
            for prod in generate2(self.rules_as_dict, node):
                pf = visit_gen(prod, visitor_make_sequence)
                if out is None:
                    out = pf
                else:
                    out = Alternation(out, pf)

            if len(self.stack) > 0:
                return self._new_rule(Rule(Symbol(self._new_name("bnf_sub")), out))
            else:
                return out
        else:
            raise ValueError(f"Subtraction with infinite productions encountered!")

    def visit_Alternation(self, node):
        is_top = len(self.stack) == 0 or self.stack[-1] != 'alt'

        self.stack.append('alt')
        node = super().visit_Alternation(node)
        self.stack.pop()

        if is_top and len(self.stack) > 0:
            # chain the alternates into a new rule
            return self._new_rule(Rule(Symbol(self._new_name("bnf_alt")),
                                       Alternation(node.expr[0], node.expr[1])))

        return node

    def visit_Optional(self, node):
        node = super().visit_Optional(node)

        #TODO: handle duplicate rules later ...?
        # expr? => '' | expr
        return self._new_rule(Rule(Symbol(self._new_name("bnf_opt")),
                                   Alternation(String(''), node.expr)))

    def visit_Concat(self, node):
        node = super().visit_Concat(node)

        #TODO: handle duplicate rules later ...?

        # expr* => concat_expr = '' | expr concat_expr
        # expr+ => concat_expr = expr | expr concat_expr
        new_name = Symbol(self._new_name("bnf_concat"))

        if node.minimum == 0:
            new_rule = Rule(new_name,
                            Alternation(String(''), Sequence(node.expr, new_name)))
        else:
            new_rule = Rule(new_name,
                            Alternation(node.expr, Sequence(node.expr, new_name)))

        return self._new_rule(new_rule)

    def visit_RuleList(self, rules):
        self.names = set([r.lhs.value for r in rules])
        self.rules_as_dict = dict([(r.lhs.value, r.rhs) for r in rules])
        self._dedup_rules = {}
        self.depth = 0
        self.stack = []
        #self.alt_top = False

        r = super().visit_RuleList(rules)
        assert len(self.stack) == 0, self.stack
        return r

def get_nodes(collection, rules, node_types, extractfn = None):
    def _visit(n):
        if isinstance(n, node_types):
            if extractfn is None:
                collection.add(n)
            else:
                extractfn(collection, n)

    visit_rules(rules, '*', _visit)

    return collection

def get_strings(rules):
    return get_nodes(set(), rules, String, lambda c, x: c.add(x.value))

def get_charclass(rules):
    return get_nodes(set(), rules, CharClass, lambda c, x: c.add(str(x)))

def test_EBNF2BNF1():
    p = EBNFParser()
    rules = p.parse('''

concat_test ::=  a*
concat_test_2 ::= b+

alt_test ::= a | b | c

alt_test_2 ::= a b (c | d )

seq_test ::= a b (c d e) f?

sub_test ::= ('a' | 'b' | 'c') - ('c')

tex_opcode_3 ::= 'tex' sep ('base' | 'level' | 'grad') sep '2dms' sep 'v4' sep tex_dtype sep 's32' | 'tex' sep ('base' | 'level' | 'grad') sep 'a2d' sep 'v4' sep tex_dtype sep 's32'
''')

    x = EBNF2BNF()
    orules = x.visit_RuleList(rules)

    for r in orules:
        print(r)

def test_dedup():
    p = EBNFParser()

    g = p.parse("""\
f_abs_opcode ::= 'abs' ((sep ftz)? sep f32 | sep f64)
f_neg_opcode ::= 'neg' ((sep ftz)? sep f32 | sep f64)
f_min_opcode ::= 'min' ((sep ftz)? sep f32 | sep f64)
f_max_opcode ::= 'max' ((sep ftz)? sep f32 | sep f64)
""")

    x = EBNF2BNF()
    x.dedup_across_rules = True
    orules = x.visit_RuleList(g)

    for r in orules:
        print(r)

def test_get_strings():
    p = EBNFParser()

    g = p.parse("""\
sep ::= '.'
ftz ::= 'ftz'
f32 ::= 'f32'
f64 ::= 'f64'
f_abs_opcode ::= 'abs' ((sep ftz)? sep f32 | sep f64)
f_neg_opcode ::= 'neg' ((sep ftz)? sep f32 | sep f64)
f_min_opcode ::= 'min' ((sep ftz)? sep f32 | sep f64)
f_max_opcode ::= 'max' ((sep ftz)? sep f32 | sep f64)
""")

    s = get_strings(g)
    print(s)
