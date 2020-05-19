#!/usr/bin/env python3
#
# ebnfast.py
#
# An AST for EBNF grammars, modeled on the W3 XML Formal Grammar
# (https://www.w3.org/TR/xml/#sec-notation)


import re

class Expression(object):
    _explicit_paren = False

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

class Char(Expression):
    precedence = 0
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return f"#x{self.value:x}"

class String(Expression):
    precedence = 0
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return repr(self.value)

class CharClass(Expression):
    precedence = 0
    def __init__(self, char_and_ranges):
        self.c_and_r = char_and_ranges

    def __str__(self):
        return f"[{''.join([str(s) for s in self.c_and_r])}]"

class Optional(Expression):
    precedence = 0

    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return f"{self.paren(self.expr)}?"

class BinOp(Expression):
    op = None

    def __init__(self, a, b):
        self.expr = [a, b]

    def __str__(self):
        return f"{self.paren(self.expr[0])}{self.op}{self.paren(self.expr[1])}"


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

    def __str__(self):
        suffix = '*' if self.minimum == 0 else '+'
        #print("STR", self.expr)
        return f"{self.paren(self.expr)}{suffix}"

class Rule(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} ::= {self.rhs}"

#TODO: comments, [ wfc: ] [ vc: ] occur at the end

class EBNFTokenizer(object):
    token = None
    match = None
    _token_stream = None

    def __init__(self, strdata):
        self.data = strdata
        self._token_stream = self.tokenize()
        self.token, self.match = next(self._token_stream)

    def tokenize(self):
        # based on the example in the re docs
        tokens = [('COMMENT', r'/\*.*\*/'),
                  ('SYMBOL', r'[A-Za-z][A-Za-z]*'),
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
                  ('WHITESPACE', r'\s+'),
                  ('MISMATCH', r'.'),
        ]

        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in tokens)

        pos = 'NORMAL' # or STR or CHARCLASS
        end_quote = None
        string = []
        for m in re.finditer(tok_regex, self.data, flags=re.M):
            token = m.lastgroup
            match = m.group()

            if pos == 'STR':
                if token != end_quote:
                    string.append(match)
                else:
                    pos = 'NORMAL'
                    end_quote = None
                    yield ('STRING', ''.join(string))
                    string = []

                continue
            elif pos == 'CHARCLASS':
                # we don't support escapes
                if token == 'RBRACK':
                    pos = 'NORMAL'
                    yield ('CHARCLASS', ''.join(string)) # let CharClass parse this?
                    string = []
                else:
                    string.append(match)
            elif pos == 'NORMAL':
                if token == 'MISMATCH':
                    raise ValueError(f"Mismatch {match}")
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

    def lookahead(self):
        return self.token

    def consume(self):
        tkn, match = self.token, self.match

        try:
            self.token, self.match = next(self._token_stream)
        except StopIteration:
            self.token, self.match = None, None

        return tkn, match

    def expect(self, token):
        tkn, match = self.consume()
        if tkn == token: return match
        assert tkn == token, f"Expecting {token}, found {tkn}"

# BNF ::= (Comment | Rule)+

# Rule ::=  Comment | Symbol '::=' Expression EOL

# Expression ::= SequenceTerm | SequenceTerm '|' Expression

# SequenceTerm ::= SubtractTerm+ #TODO: this requires reasoning about what will terminate a sequence term

# SubtractTerm ::= ConcatTerm ('-' ConcatTerm)*

# ConcatTerm ::= Term ( '*' | '+' | '?' )?

# Term ::= Symbol | Hexadecimal | CharClass | StringLiteral | '(' Expression ')'

class EBNFParser(object):
    def parse(self, ebnf, token_stream = None):
        if token_stream is None:
            token_stream = EBNFTokenizer(ebnf)

        out = []
        while True:
            tkn, match = token_stream.consume()
            if tkn == "COMMENT": # only allow comments at top level?
                continue
            elif tkn == "SYMBOL":
                lhs = match
                token_stream.expect('RULEDEF')
                rhs = self.parse_expr(token_stream)
                out.append(Rule(lhs, rhs))
            elif tkn is None:
                break
            elif tkn == 'EOL':
                continue # maybe expect at end of rule?
            else:
                assert False, f"Unexpected token {tkn}"

        return out

    def parse_expr(self, token_stream):
        out = self.parse_sequence(token_stream)
        while True:
            ltkn = token_stream.lookahead()
            if ltkn == 'EOL' or ltkn is None: break
            if ltkn == 'ALT':
                token_stream.consume()
                # treat | as left associative?
                out = Alternation(out, self.parse_expr(token_stream))
            elif ltkn == 'RPAREN':
                break
            else:
                assert False, f"Unexpected token {ltkn}"

        return out

    def parse_sequence(self, token_stream):
        out = self.parse_subtract(token_stream)
        while token_stream.lookahead() not in ('ALT', 'EOL', 'RPAREN', None):
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
            print(f"Unmatched {tkn}/{match} when parsing term")


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


