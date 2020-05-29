#!/usr/bin/env python3
#
# ebnfast.py
#
# An AST for EBNF grammars, modeled on the W3 XML Formal Grammar
# (https://www.w3.org/TR/xml/#sec-notation)


import re

class Expression(object):
    _explicit_paren = False
    children = []

    def paren(self, expr):
        # lower precedence value indicates higher precedence
        if (not expr._explicit_paren) and (expr.precedence <= self.precedence):
            return str(expr)
        else:
            return f"({expr})"

class Symbol(Expression):
    precedence = 0
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return self.value

    def graph(self):
        return (str(self), [])

class Char(Expression):
    precedence = 0
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return f"#x{self.value:x}"

    def graph(self):
        return (str(self), [])

class String(Expression):
    precedence = 0
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return "'" + self.value + "'"

    def graph(self):
        return (str(self), [])

# really an enum?
class CharClass(Expression):
    precedence = 0
    def __init__(self, char_and_ranges):
        self.c_and_r = char_and_ranges
        self.invert = self.c_and_r[0] == '^'

        if self.invert:
            self.c_and_r = self.c_and_r[1:]

        self._parse(self.c_and_r)

    def _parse(self, s):
        tokens = [('HEX', r'#x[A-Fa-f0-9]+'),
                  ('MINUS', r'-'),
                  ('CHAR', r'.'),]

        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in tokens)

        enum = []
        for m in re.finditer(tok_regex, s):
            token = m.lastgroup
            match = m.group()

            if token == 'HEX':
                enum.append(int(match[2:], 16))
            elif token == 'MINUS':
                enum.append(match)
            elif token == 'CHAR':
                enum.append(ord(match))


        i = 0
        while i < len(enum):
            if enum[i] == "-":
                assert (i - 1 >= 0) and (i + 1 < len(enum)), f"Incorrect syntax in {self.c_and_r}"
                enum[i-1] = (enum[i-1], enum[i+1])
                del enum[i+1]
                del enum[i]
            else:
                i += 1

        self.enum = enum

    def iter(self):
        # this should be invert of Char?
        if self.invert: raise NotImplementedError(f"Can't iterate over [^...] yet")

        for x in self.enum:
            if isinstance(x, tuple):
                for c in range(x[0], x[1] + 1):
                    yield chr(c)
            else:
                yield chr(x)

    def __str__(self):
        return f"[{'^' if self.invert else ''}{self.c_and_r}]"

    def graph(self):
        return (str(self), [])

class Optional(Expression):
    precedence = 0

    def __init__(self, expr):
        self.expr = expr
        self.children = [self.expr]

    def __str__(self):
        return f"{self.paren(self.expr)}?"

    def graph(self):
        return ("?", self.children)

class BinOp(Expression):
    op = None

    def __init__(self, a, b):
        self.expr = [a, b]
        self.children = self.expr

    def __str__(self):
        x1 = self.paren(self.expr[0])
        x2 = self.paren(self.expr[1])
        o = self.op
        #return f"{x1}{self.op}{x2}"

        if self.op == ' | ':
            if len(x1) > 50:
                o = '\n        | '

        return f"{x1}{o}{x2}"

    def graph(self):
        return (self.op.strip(), self.children)

class Sequence(BinOp):
    precedence = 0
    op = ' '

class Alternation(BinOp):
    precedence = 2
    op = ' | '

class Subtraction(BinOp):
    precedence = 3 #???
    op = ' - '

# handles both one or more, zero or more
class Concat(Expression):
    precedence = 0

    def __init__(self, expr, minimum: int = 0):
        self.expr = expr
        self.minimum = minimum
        self.children = [self.expr]

    def __str__(self):
        suffix = '*' if self.minimum == 0 else '+'
        #print("STR", self.expr)
        return f"{self.paren(self.expr)}{suffix}"

    def graph(self):
        return ('*' if self.minimum == 0 else '+', self.children)

class Rule(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.children = [self.lhs, self.rhs]

    def __str__(self):
        return f"{self.lhs} ::= {self.rhs}"

    def graph(self):
        return ('::=', self.children)


def visualize_ast(root, edgelist, rule_dict = None):
    key = id(root)

    if key in edgelist:
        return

    print(root)
    label, children = root.graph()

    if isinstance(root, Symbol) and rule_dict and label in rule_dict:
        children = [rule_dict[label]]

    edgelist[key] = [label] + [id(c) for c in children]

    for c in children:
        visualize_ast(c, edgelist, rule_dict)

def generate_dot(graph):
    yield "digraph {"
    yield "node[shape=none];"
    for n in graph:
        lbl = graph[n][0]
        if not lbl or lbl[0] != '"':
            lbl = '"' + lbl + '"'

        yield f"{n} [label={lbl}]"
        for e in graph[n][1:]:
            yield f"{n} -> {e}"

    yield "}"

def generate_graph(root, output_routine, rule_dict):
    edgelist = {}
    visualize_ast(root, edgelist, rule_dict)
    return output_routine(edgelist)

#TODO: comments, [ wfc: ] [ vc: ] occur at the end

class ParseError(object):
    def error(self, tok, message):
        import sys
        print(f"{tok.err_scoord}: {message}", file=sys.stderr)
        print("    " + tok.err_line, file=sys.stderr)
        print(" "*(tok.err_coord[1][0]+4) +"^"*(tok.err_coord[1][1] - tok.err_coord[1][0]), file=sys.stderr)
        raise ValueError(message)

class EBNFTokenizer(object):
    token = None
    match = None
    _token_stream = None

    def __init__(self, strdata, err):
        self.data = strdata
        self._token_stream = self.tokenize()
        self.token, self.match = next(self._token_stream)
        self.err = err

    # convenience
    def error(self, message):
        self.err.error(self, message)

    def tokenize(self):
        # based on the example in the re docs
        tokens = [('COMMENT', r'/\*.*\*/'),
                  ('SYMBOL', r'[A-Za-z][A-Za-z0-9_]*'), # numbers and underscore are an extension from W3?
                  ('RULEDEF', r'::='),
                  ('HEX', r'#x[A-Fa-f0-9]+'),
                  ('DBLQUOTE', r'"'),
                  ('QUOTE', r"'"),
                  ('LPAREN', r'\('),
                  ('RPAREN', r'\)'),
                  ('LBRACK', r'\['),
                  ('RBRACK', r'\]'),
                  ('STAR', r'\*'),
                  ('PLUS', r'\+'),
                  ('MINUS', r'-'),
                  ('OPT', r'\?'),
                  ('ALT', r'\|'),
                  ('EOL', r'\n'),
                  ('WHITESPACE', r'[ \t]+'),
                  ('MISMATCH', r'.'),
        ]

        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in tokens)

        pos = 'NORMAL' # or STR or CHARCLASS
        end_quote = None
        string = []
        for lno, l in enumerate(self.data.split('\n')):
            self.line = l

            for m in re.finditer(tok_regex, l + '\n', flags=re.M):
                token = m.lastgroup
                match = m.group()
                self.coord = (lno+1, m.span())
                self.scoord = f'{self.coord[0]}:{self.coord[1][0]}-{self.coord[1][1]}'

                if pos == 'STR':
                    if token != end_quote:
                        string.append(match)
                    else:
                        pos = 'NORMAL'
                        end_quote = None
                        yield ('STRING', ''.join(string))
                        string = []
                elif pos == 'CHARCLASS':
                    # we don't support escapes
                    if token == 'RBRACK':
                        pos = 'NORMAL'
                        yield ('CHARCLASS', ''.join(string)) # CharClass parses this
                        string = []
                    else:
                        string.append(match)
                elif pos == 'NORMAL':
                    if token == 'MISMATCH':
                        self._update_err_pos()
                        self.error(f"Unrecognized token '{match}'")
                    elif token == 'QUOTE' or token == 'DBLQUOTE':
                        pos = 'STR'
                        end_quote = token
                    elif token == 'WHITESPACE':
                        # Sequence?
                        pass
                    elif token == 'LBRACK':
                        pos = 'CHARCLASS'
                    else:
                        yield (token, match)

            if pos != 'NORMAL':
                self._update_err_pos()
                self.error(f"Encountered end-of-line when scanning {pos}")


    def _update_err_pos(self):
        self.err_coord = self.coord
        self.err_line = self.line
        self.err_scoord = self.scoord

    def lookahead(self):
        return self.token

    def consume(self):
        self._update_err_pos()
        tkn, match = self.token, self.match

        try:
            self.token, self.match = next(self._token_stream)
        except StopIteration:
            self.token, self.match = None, None

        return tkn, match

    def expect(self, token):
        tkn, match = self.consume()
        if tkn == token:
            return match
        else:
            self.error(f"Expecting {token}, found {tkn}")

# BNF ::= (Comment | Rule)+

# Rule ::=  Comment | Symbol '::=' Expression EOL

# Expression ::= SequenceTerm | SequenceTerm '|' Expression

# SequenceTerm ::= SubtractTerm+ #TODO: this requires reasoning about what will terminate a sequence term

# SubtractTerm ::= ConcatTerm ('-' ConcatTerm)*

# ConcatTerm ::= Term ( '*' | '+' | '?' )?

# Term ::= Symbol | Hexadecimal | CharClass | StringLiteral | '(' Expression ')'

class EBNFParser(object):
    def parse(self, ebnf, token_stream = None, as_dict = False):
        if token_stream is None:
            err = ParseError()
            token_stream = EBNFTokenizer(ebnf, err)

        rule_names = set()
        out = []
        while True:
            tkn, match = token_stream.consume()
            if tkn == "COMMENT":
                continue
            elif tkn == "SYMBOL":
                lhs = match

                if lhs in rule_names:
                    token_stream.error(f"Duplicate rule '{lhs}'")
                else:
                    rule_names.add(lhs)

                token_stream.expect('RULEDEF')
                rhs = self.parse_expr(token_stream)
                out.append(Rule(Symbol(lhs), rhs))
            elif tkn is None:
                break
            elif tkn == 'EOL':
                continue # maybe expect at end of rule?
            elif tkn == 'ALT':
                # this is an extension
                if len(out) == 0:
                    token_stream.error(f"Continuation found, but previous line was not a rule")

                rhs = self.parse_expr(token_stream)
                prev_rule = out[-1]
                prev_rule.rhs = Alternation(prev_rule.rhs, rhs)
            else:
                token_stream.error(f"Unexpected token {tkn} ({match}) when parsing rules")

        if as_dict:
            dout = dict([(r.lhs.value, r.rhs) for r in out])
            return dout
        else:
            return out

    def parse_expr(self, token_stream):
        out = self.parse_sequence(token_stream)
        while True:
            ltkn = token_stream.lookahead()
            if ltkn == 'EOL' or ltkn is None or ltkn == 'COMMENT': break
            if ltkn == 'ALT':
                token_stream.consume()
                # treat | as left associative?
                out = Alternation(out, self.parse_expr(token_stream))
            elif ltkn == 'RPAREN':
                break
            else:
                token_stream.error(f"Unexpected token {ltkn} when parsing expression")

        return out

    def parse_sequence(self, token_stream):
        out = self.parse_subtract(token_stream)
        while token_stream.lookahead() not in ('ALT', 'EOL', 'RPAREN', 'COMMENT', None):
            out = Sequence(out, self.parse_subtract(token_stream))

        return out

    def parse_subtract(self, token_stream):
        out = self.parse_concat(token_stream)

        while token_stream.lookahead() == 'MINUS':
            token_stream.consume()
            out = Subtraction(out, self.parse_concat(token_stream))

        return out

    def parse_concat(self, token_stream):
        term = self.parse_term(token_stream)
        tkn = token_stream.lookahead()
        # this currently allows whitespace between term and */+/?
        if tkn == 'OPT':
            token_stream.consume()
            return Optional(term)
        elif tkn == 'STAR':
            token_stream.consume()
            return Concat(term, 0)
        elif tkn == 'PLUS':
            token_stream.consume()
            return Concat(term, 1)
        else:
            return term

    def parse_term(self, token_stream):
        tkn, match = token_stream.consume()

        if tkn == 'HEX':
            return Char(int(match[2:], 16))
        elif tkn == 'STRING':
            return String(match)
        elif tkn == 'CHARCLASS':
            return CharClass(match)
        elif tkn == 'SYMBOL':
            return Symbol(match)
        elif tkn == 'LPAREN':
            expr = self.parse_expr(token_stream)
            token_stream.expect('RPAREN')
            expr._explicit_paren = True
            return expr
        else:
            token_stream.error(f"Unexpected token {tkn} when parsing term")

def test_parser_1():
    p = EBNFParser()

    x = p.parse('''
BNF ::= (Rule | Comment)+

Rule ::=  Comment | Symbol '::=' Expression

Expression ::=  Hexadecimal | CharClass | StringLiteral | Expression '?' | ( Expression '|' Expression ) | '(' Expression ')' | Expression '-' Expression | Expression '+' | Expression '*'

''')
    for xx in x:
        print(xx)


def test_charclass():
    p = EBNFParser()

    x = p.parse('''
CC ::= [A-Za-z]

CCI ::= [^A-Za-z]
''')
    for xx in x:
        print(xx)


def test_realworld():
    p = EBNFParser()

    x = p.parse('''
Char	   ::=   	#x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]

S	   ::=   	(#x20 | #x9 | #xD | #xA)+

Comment	   ::=   	'<!--' ((Char - '-') | ('-' (Char - '-')))* '-->'

content	   ::=   	CharData? ((element | Reference | CDSect | PI | Comment) CharData?)*
''')
    for xx in x:
        print(xx)

def test_graph_gen():
    p = EBNFParser()

    x = p.parse("content	   ::=   	CharData? ((element | Reference | CDSect | PI | Comment) CharData?)*")

    g = generate_graph(x[0], generate_dot)
    with open("/tmp/g.dot", "w") as f:
        for l in g:
            print(l, file=f)

def test_continuations():

    grammar = """
    test ::= 'xyz' | 'abc'
    test2 ::= 'abc'
     | 'xyz' /* xyz */
     | test  /* abc */

"""
    p = EBNFParser()
    x = p.parse(grammar, as_dict = True)
    print(x)

def test_bug():
    # this was a hard to trigger bug with whitespace including \n under certain circumstances
    grammar = """mma_1_type ::= f16 | f32
mma_1_layout ::= 'row' | 'col'
mma_prefix ::= 'mma' sep 'sync' sep 'aligned'   
mma_opcode_1 ::= mma_prefix sep 'm8n8k4' sep mma_1_layout sep mma_1_layout sep mma_1_type sep f16 sep f16 sep mma_1_type
mma_opcode_2 ::= mma_prefix sep 'm16n8k8' sep 'row' sep 'col' sep mma_1_type sep f16 sep f16 sep mma_1_type
mma_opcode ::= mma_opcode_1 | mma_opcode_2
"""
    p = EBNFParser()
    x = p.parse(grammar, as_dict = True)
    print(x)

