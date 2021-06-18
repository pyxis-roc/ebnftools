from .. import tokens
from ... import ebnfast
import re
import textwrap

LEX_PROLOGUE = """#!/usr/bin/env python3
import ply.lex as lex
import sys
{leximports}
"""

LEX_EPILOGUE = """
lexer = lex.lex(optimize=1)

if __name__ == '__main__':
    lex.runmain()
"""


YACC_PROLOGUE = """#!/usr/bin/env python3
import ply.yacc as yacc
from {lexer} import tokens, lexer
from ebnftools.convert.ply import utils
import sys
{imports}

start = '{start}'
"""

YACC_EPILOGUE = """
parser = yacc.yacc()

if __name__ == '__main__':
    while True:
        try:
            s = input('line> ')
            print("your input was '", s, "'")
        except EOFError:
            break
        if not s: continue
        result = parser.parse(s)
        print(result)
        if result is not None and hasattr(result, 'args'):
            d = utils.vis_parse_tree(result)
            ds = "\\n".join(d)
            with open('parsetree.dot', 'w') as f:
                print(ds, file=f)

        print(utils.visit_abstract(result))


"""

class IndirectRule:
    def __init__(self, tmap, fn):
        self.tmap = tmap
        self.fn = fn

    def get_code(self):
        out = []
        out.append(self.tmap)
        out.append(self.fn)
        return out

class RegexpRule:
    def __init__(self, rule):
        self.rule = rule

    def get_code(self):
        return [self.rule]

class FunctionRule:
    def __init__(self, fn):
        self.fn = fn

    def get_code(self):
        return [self.fn]

class LexerGen(object):
    def __init__(self, tokenreg, action_tokens = None, lexermod = None, modpath=None):
        self.treg = tokenreg
        self.indirect_tokens = {}
        self.ignore_tokens = set()
        self.action_tokens = action_tokens if action_tokens is not None else {}
        self.lexermod = lexermod
        self.modpath = modpath if modpath else ''
        self.gen = {}
        self.track_lines = True
        self.t_error = None

    def gen_rule(self, tkn, rule):
        assert tkn not in self.gen, f"Duplicate code generation"
        self.gen[tkn] = rule

    def add_indirect(self, token, indirect_token):
        self.indirect_tokens[token] = indirect_token

    def add_ignore(self, token):
        self.ignore_tokens.add(token)

    def _generate_tokens(self):
        sep = ",\n    "
        out = f"""tokens = ({sep.join([repr(s) for s in self.treg.tokens])})\n"""
        return out

    def _get_py_re_cc(self, v):
        def mkchar(c):
            if c < 32 or c > 127:
                return f"\\x{c:02x}"
            else:
                return chr(c)

        out = []
        for e in v.enum:
            if isinstance(e, tuple):
                # range
                out.append(f"{mkchar(e[0])}-{mkchar(e[1])}")
            elif e == ord("["):
                out.append("\\" + chr(e))
            else:
                out.append(mkchar(e))

        out = "".join(out)
        out = f"[{'^' if v.invert else ''}{out}]"

        return out

    def _get_re(self, v):
        '''Returns a string literal for a particular Tkn type'''
        if isinstance(v, (tokens.TknLiteral, tokens.TknRegExp)):
            s = str(v)[1:-1]
            # this also handles escaping for re.VERBOSE
            if isinstance(v, tokens.TknLiteral):
                s = re.escape(s)
            s = repr(s)
        elif isinstance(v, tokens.TknCharClass):
            s = repr(self._get_py_re_cc(v))
        else:
            raise NotImplementedError(f"Unknown Token class {type(v)}")

        return s

    def _gen_indirect_rules(self):
        # dictionaries per rule

        indirect_tokens = set(self.indirect_tokens.values())

        buckets = dict([(t, []) for t in indirect_tokens])

        for k in self.indirect_tokens:
            buckets[self.indirect_tokens[k]].append(k)

        for t in buckets:
            out = []
            for k in buckets[t]:
                assert self.indirect_tokens[k] == t
                assert isinstance(self.treg.n2v[k], tokens.TknLiteral), f"Only literals permitted in indirect tokens"
                value = self.treg.n2v[k]
                out.append(f"{value}: '{k}'")

            out = ',\n'.join(out)

            if t in self.action_tokens:
                # we only invoke the action if t's type is preserved by the map
                action = f"if t.type == {repr(t)}: t.value = {self.action_tokens[t]}(t.value)"
            else:
                action = ""

            fn = f"""
def t_{t}(t):
    {self._get_re(self.treg.n2v[t])}
    t.type = indirect_{t}.get(t.value, '{t}')
    {action}
    return t
"""
            tdict = f"indirect_{t} = {{{out}}}"
            self.gen_rule(t, IndirectRule(tdict, fn))

        return

    def _gen_simple_rules(self):
        simple_tokens = self.treg.tokens - set(self.indirect_tokens.keys()).union(self.indirect_tokens.values()).union(self.ignore_tokens)

        token_types = {'literals': [], 're': []}

        for t in self.treg.read_order:
            if t not in simple_tokens: continue

            v = self.treg.n2v[t]
            re = self._get_re(v)

            if isinstance(v, tokens.TknLiteral):
                token_types['literals'].append((t, re))
            else:
                token_types['re'].append((t, re))

        for (t, re) in token_types['re']:
            if t not in self.action_tokens:
                self.gen_rule(t, RegexpRule(f't_{t} = {re}'))
            else:
                fn = f"""
def t_{t}(t):
    {re}
    t.value = {self.action_tokens[t]}(t.value)
    return t
"""
                self.gen_rule(t, FunctionRule(fn))

        # prioritize string literals over regular expressions by
        # converting them to functions. Indirect rules are not
        # affected.
        token_types['literals'].sort(key=lambda x: len(x), reverse=True)
        for (t, re) in token_types['literals']:
            f = f"""
def t_{t}(t):
    {re}
    return t
"""
            self.gen_rule(t, FunctionRule(f))

        return

    def _generate_ignore_rules(self):
        for t in self.ignore_tokens:
            re = self._get_re(self.treg.n2v[t])
            f = f"""
def t_{t}(t):
   {re}
   return None
"""
            self.gen_rule(t, FunctionRule(f))

        return

    def _generate_rules(self):
        self.gen = {}
        self._generate_ignore_rules()
        self._gen_simple_rules()
        self._gen_indirect_rules()

        out = []
        if self.track_lines:
            out.append(f"""
def t_newline(t):
   r"\\n+"
   t.lexer.lineno += len(t.value)
""")
        if self.t_error:
            out.append(self.t_error)

        for t in self.treg.read_order:
            if t in self.gen:
                out.extend(self.gen[t].get_code())

        code = "\n".join(out)
        return code

    def get_lexer(self):
        leximports = f"from {self.modpath}{self.lexermod} import *" if self.lexermod else ""
        out = LEX_PROLOGUE.format(leximports=leximports) + self._generate_tokens() + self._generate_rules() + LEX_EPILOGUE
        return out

class ActionGen(object):
    def get_action(self, rule):
        raise NotImplementedError

    def get_module(self, rule):
        return ""

class PassActionGen(ActionGen):
    def get_action(self, rule):
        return "pass"

    def get_module(self):
        return ""

class CTActionGen(ActionGen):
    """Generates a concrete parse tree"""

    def __init__(self, abstract = False):
        self.classes = {}
        self.abstract = abstract # generate an AST by calling abstract()

    def _get_rule_length(self, rhs):
        if isinstance(rhs, ebnfast.Symbol):
            return [1]
        elif isinstance(rhs, ebnfast.String):
            assert rhs.value == ''
            return [0]
        elif isinstance(rhs, ebnfast.Sequence):
            y = self._get_rule_length(rhs.expr[0])[0] #TODO: weird
            return [y + x for x in self._get_rule_length(rhs.expr[1])]
        elif isinstance(rhs, ebnfast.Alternation):
            o = self._get_rule_length(rhs.expr[0])
            o.extend(self._get_rule_length(rhs.expr[1]))
            return o
        else:
            raise NotImplementedError(f"Don't know how to handle {rhs}")

    def _is_opt_rule(self, rule):
        # TODO: actually check for an empty string
        if isinstance(rule.rhs, ebnfast.Alternation):
            if isinstance(rule.rhs.expr[0], ebnfast.String):
                if rule.rhs.expr[0].value == '':
                    return True

        return False

    def get_action(self, rule):
        rl = self._get_rule_length(rule.rhs)
        rls = sorted(set(rl))

        rn = str(rule.lhs)

        if rn not in self.classes:
            self.classes[rn] = (f"a_{rn}", rls, rule)

        out = []
        action = f"{self.classes[rn][0]}(p)"
        if self.abstract:
            action += ".abstract()"

        if self._is_opt_rule(rule):
            # non-empty matches have a len of 2
            if isinstance(rule.rhs.expr[1], ebnfast.Symbol):
                # format is 'symbol?', where symbol is a top-level
                # rule, so just pass it on instead of creating an
                # concrete syntree node for this.
                out.append(f"p[0] = None if (len(p) == 1) else p[1]")
            else:
                out.append(f"p[0] = None if (len(p) == 1) else {action}")
        else:
            out.append(f"p[0] = {action}")

        return "\n".join(out)

    def _get_block(self, indent, *lines):
        return textwrap.indent("\n".join(lines), '    '*indent)

    def _gen_arglen_match(self, l):
        def _match_one(rulelen):
            if rulelen == 0:
                return ["pass"]
            else:
                return [f"self.args[{i}] = p[{i+1}]" for i in range(rulelen)]

        out = []
        if len(l) == 1:
            out = _match_one(l[0])
        else:
            out.append(f"if len(p) == {l[0]+1}:")
            out.append(self._get_block(1, *_match_one(l[0])))

            for ll in l[1:]:
                out.append(f"elif len(p) == {ll+1}:")
                out.append(self._get_block(1, *_match_one(ll)))

        return "\n".join(out)


    def get_module(self):
        mod = []

        for c in self.classes:
            cn, args, rule = self.classes[c]

            out = [f"class {cn}:"]
            out.append("    def __init__(self, p):")
            out.append(self._get_block(2,
                                       f"self.args = [None]*{max(args)}",
                                       self._gen_arglen_match(args)))
            out.append("")
            out.append("    def __str__(self):")
            out.append(self._get_block(2,
                                       "v = ', '.join([str(s) for s in self.args])",
                                       f"return f'{cn}({{v}})'"))
            out.append("    __repr__ = __str__")
            out.append("    def abstract(self):")
            out.append(self._get_block(2, "return self"))
            out.append("")

            mod.extend(out)

        return "\n".join(mod)


class ParserGen(object):
    def __init__(self, treg, bnf, start_symbol, actiongen = None, handlermod = None, modpath = None):
        self.bnf = bnf
        self.treg = treg
        self.start = start_symbol
        self.bnfgr = dict([(r.lhs.value, r.rhs) for r in self.bnf])
        self.actiongen = actiongen if actiongen is not None else PassActionGen()
        self.handlermod = handlermod
        self.modpath = modpath if modpath else ''
        self.p_error = None

    def _get_reachable_symbols(self):
        def _visit(n):
            if isinstance(n, ebnfast.Symbol):
                if n.value not in self.treg.tokens:
                    if n.value not in reachable:
                        reachable.add(n.value)
                        dfs.insert(0, n.value)

        reachable = set([self.start])
        dfs = [self.start]
        ebnfast.visit_rule_dict(self.bnfgr, self.bnfgr[self.start], _visit)
        return dfs

    def _flatten(self, rhs):
        if isinstance(rhs, ebnfast.String):
            assert rhs.value == '', f"Non-empty string literals ({rhs}) in {rhs} not supported in PLY BNF"
            return []
        elif isinstance(rhs, ebnfast.Sequence):
            o = self._flatten(rhs.expr[0])
            o.extend(self._flatten(rhs.expr[1]))
            #o = [' '.join(oo) for oo in o]
            return [' '.join(o)]
        elif isinstance(rhs, ebnfast.Alternation):
            o = [' '.join(self._flatten(rhs.expr[0]))]
            o.extend(self._flatten(rhs.expr[1]))
            return o
        elif isinstance(rhs, ebnfast.Symbol):
            return [rhs.value]
        else:
            raise NotImplementedError(f"Don't know how to flatten {rhs}")

    def _get_rule(self, r):
        assert isinstance(r.rhs, (ebnfast.Sequence, ebnfast.Alternation, ebnfast.Symbol,
                                  ebnfast.Char, ebnfast.String)), f"Unknown rule type: {type(r.rhs)}"

        if isinstance(r.rhs, ebnfast.Symbol):
            return f"'{r.lhs.value} : {r.rhs}'"
        else:
            try:
                rhs = self._flatten(r.rhs)
                spaces = "\n" + " "*(len(r.lhs.value)+2+4+2) + "| " # spaces not needed, but looks better
                rhs = spaces.join(rhs)
                rhs = f"'''{r.lhs.value} : {rhs}'''"
            except AssertionError:
                print("Error when constructing rule {r.lhs}")
                raise

        return rhs

    def get_parse_rules(self):
        template = """
# {original}
def p_{nonterminal}(p):
    {bnfrule}
{action}
"""
        out = []
        for s in self._get_reachable_symbols():
            tv = {}
            rl = ebnfast.Rule(ebnfast.Symbol(s), self.bnfgr[s])
            tv['original'] = str(self.bnfgr[s])
            tv['nonterminal'] = s
            tv['bnfrule'] = self._get_rule(rl)
            tv['action'] = textwrap.indent(self.actiongen.get_action(rl), ' '*4)
            out.append(template.format(**tv))

        return "\n\n".join(out)

    def get_parser(self, lexer):
        if self.handlermod:
            imports = [f'from {self.modpath}{self.handlermod} import *']
        else:
            imports = []

        imports = '\n'.join(imports)

        p_error = self.p_error if self.p_error is not None else ""

        out = YACC_PROLOGUE.format(lexer=lexer, start=self.start, imports=imports) + p_error + self.get_parse_rules() + YACC_EPILOGUE
        return out

    def get_action_module(self):
        return self.actiongen.get_module()
