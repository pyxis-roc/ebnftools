from .. import tokens
from ... import ebnfast
import re

LEX_PROLOGUE = """#!/usr/bin/env python3
import ply.lex as lex
"""

LEX_EPILOGUE = """
lexer = lex.lex()

if __name__ == '__main__':
    lex.runmain()
"""


YACC_PROLOGUE = """#!/usr/bin/env python3
import ply.yacc as yacc
from {lexer} import tokens

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
"""

class LexerGen(object):
    def __init__(self, tokenreg):
        self.treg = tokenreg
        self.indirect_tokens = {}
        self.ignore_tokens = set()

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

        out2 = []
        for t in buckets:
            out = []
            for k in buckets[t]:
                assert self.indirect_tokens[k] == t
                assert isinstance(self.treg.n2v[k], tokens.TknLiteral), f"Only literals permitted in indirect tokens"
                value = self.treg.n2v[k]
                out.append(f"{value}: '{k}'")

            out = ',\n'.join(out)
            out2.append(f"indirect_{t} = {{{out}}}")

            fn = f"""
def t_{t}(t):
    {self._get_re(self.treg.n2v[t])}
    t.type = indirect_{t}.get(t.value, '{t}')
    return t
"""
            out2.append(fn)

        return "\n".join(out2) + "\n\n"

    def _gen_simple_rules(self):
        simple_tokens = self.treg.tokens - set(self.indirect_tokens.keys()).union(self.indirect_tokens.values()).union(self.ignore_tokens)

        token_types = {'literals': [], 're': []}

        out = []
        for t in simple_tokens:
            v = self.treg.n2v[t]
            re = self._get_re(v)

            if isinstance(v, tokens.TknLiteral):
                token_types['literals'].append((t, re))
            else:
                token_types['re'].append((t, re))

        for (t, re) in token_types['re']:
            out.append(f't_{t} = {re}')

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
            out.append(f)

        return "\n".join(out) + "\n\n"

    def _generate_ignore_rules(self):
        out = []
        for t in self.ignore_tokens:
            re = self._get_re(self.treg.n2v[t])
            f = f"""
def t_{t}(t):
   {re}
   return None
"""
            out.append(f)

        return "\n".join(out)  + "\n\n"

    def _generate_rules(self):
        return self._generate_ignore_rules() + self._gen_simple_rules() + self._gen_indirect_rules()

    def get_lexer(self):
        out = LEX_PROLOGUE + self._generate_tokens() + self._generate_rules() + LEX_EPILOGUE
        return out

class ParserGen(object):
    def __init__(self, treg, bnf, start_symbol):
        self.bnf = bnf
        self.treg = treg
        self.start = start_symbol
        self.bnfgr = dict([(r.lhs.value, r.rhs) for r in self.bnf])

    def _get_reachable_symbols(self):
        def _visit(n):
            if isinstance(n, ebnfast.Symbol):
                if n.value not in self.treg.tokens:
                    if n.value not in reachable:
                        reachable.add(n.value)
                        dfs.insert(0, n.value)

        reachable = set()
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
    pass
"""
        out = []
        for s in self._get_reachable_symbols():
            tv = {}
            tv['original'] = str(self.bnfgr[s])
            tv['nonterminal'] = s
            tv['bnfrule'] = self._get_rule(ebnfast.Rule(ebnfast.Symbol(s), self.bnfgr[s]))
            out.append(template.format(**tv))

        return "\n\n".join(out)

    def get_parser(self, lexer):
        out = YACC_PROLOGUE.format(lexer=lexer, start=self.start) + self.get_parse_rules() + YACC_EPILOGUE
        return out
