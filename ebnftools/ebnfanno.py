#!/usr/bin/env python3
#
# ebnfanno.py
#
# Allow `annotations' in EBNF specifications.
#
# Currently, these annotations are used to specify:
#
# 1. Constraints, using @CONSTRAINT, and @CONSTRAINT_DEBUG
# 2. Actions, using @ACTION
# 3. AST generation?
#
# The syntax is a subset of Lisp, primarily for ease of parsing and
# translation.
#
# Annotations are independent of the EBNF. They are parsed and
# extracted out before the EBNF.  This matters largely for comment
# processing. Commented annotations must begin with /*@. Since the
# EBNF parser does not support multi-line comments, this is not a
# problem in practice, yet.
#
# This also means that line numbers are all wonky in the error messages from the EBNF Parser.
# Ultimately, this should be a unified parser?

import re

# reuse AST nodes, but pretend they belong here (i.e. users should use ebnfanno.Symbol)
try:
    from .ebnfast import Symbol, String, Coord
except ImportError:
    from ebnfast import Symbol, String, Coord

class Number(object):
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return str(self.value)

# based on smt2ast parser
class SExprList(object):
    def __init__(self, *args):
        self.value = args
        self.coord = None

    def __str__(self):
        return f"({' '.join([str(s) for s in self.value])})"


def Anno(name, value):
    return SExprList(Symbol(name) if isinstance(name, str) else name, *value)

# annotations start with a @ symbol as the first character of the line
# They are followed by by S-expression list @(), that can contain the following tokens:
#
#  Symbols
#  White space
#  Parentheses
#  Strings
#
# This parser should be called with the buffer pointing to the
# parenthesis after @. It will terminate when it encounters the closing ).
#
# The annotation can span multiple lines.
class AnnoParser(object):
    def tokenize(self, data, dataiter):
        # based on the example in the re docs
        tokens = [('STRING1', r'"[^"]*"'),
                  ('STRING2', r"'[^']*'"),
                  ('LPAREN', r'\('),
                  ('RPAREN', r'\)'),
                  ('NUMBER', r'[0-9]+'),
                  ('SYMBOL', r'[A-Za-z_][A-Za-z0-9_]*'),
                  ('WHITESPACE', r'\s+'),
                  ('MISMATCH', r'.'),]

        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in tokens)
        parens = 0

        try:
            while True:
                self.consumed = 0 # track how much of a line we've consumed

                for m in re.finditer(tok_regex, data, flags=re.M):
                    token = m.lastgroup
                    match = m.group()

                    self.consumed += len(match)
                    if token == 'MISMATCH':
                        self.consumed -= len(match)
                        raise ValueError(f"Mismatch '{match}' (when parsing {data})")
                    elif token == 'WHITESPACE':
                        pass
                    else:
                        if token == 'LPAREN':
                            parens += 1
                        elif token == 'RPAREN':
                            parens -= 1

                        yield (token, match)

                if parens == 0: break
                data = next(dataiter)

        except StopIteration:
            raise ValueError(f"Unexpected end of input when tokenizing")

    def parse(self, llstr, dataiter, linepos, token_stream = None):
        start = linepos.curline
        if token_stream is None:
            token_stream = self.tokenize(llstr, dataiter)

        out = []
        try:
            while True:
                tkn, match = next(token_stream)
                if tkn == "RPAREN":
                    return SExprList(*out)
                    # TODO: possibly add a validator here?
                elif tkn == "LPAREN":
                    out.append(self.parse(llstr, dataiter, linepos, token_stream))
                elif tkn == "SYMBOL":
                    out.append(Symbol(match))
                elif tkn == "NUMBER":
                    out.append(Number(int(match, 10)))
                elif tkn == "STRING1" or tkn == "STRING2":
                    out.append(String(match[1:-1]))
                else:
                    raise NotImplementedError(f"Unknown token {tkn} '{match}'")
        except StopIteration:
            #raise ValueError("Ran out of input when parsing SExpr")
            pass

        if len(out) == 1:
            if isinstance(out[0], SExprList):
                out[0].coord = Coord(start, start, linepos.curline)

            return out[0]
        else:
            raise ValueError(f"Parse resulted in multiple items! {out}")

class LineConsumer(object):
    def __init__(self, linearray, start = 0):
        self.la = linearray
        self.start = start
        self.next_start = 0

    @property
    def curline(self):
        return self.next_start+1 # use 1-based

    def lines(self):
        for l in range(self.start, len(self.la)):
            self.next_start = l
            yield self.la[l]

def parse_annotated_grammar(gr):
    #data = [l + '\n' for l in gr.split('\n')]
    data = gr.split('\n')

    lco = LineConsumer(data)
    lc = lco.lines()
    p = AnnoParser()

    ebnf = []
    anno = []
    try:
        while True:
            l = next(lc)
            if l and l[0] == '@': # forces first character to be '@', no leading whitespace allowed

                # we discard everything on the same line after the ending ')'
                anno.append(p.parse(l[1:], lc, lco))
            else:
                ebnf.append(l)
    except StopIteration:
        pass

    return ebnf, anno

def test_parser():
    data2 = """@(CONSTRAINT (tex_opcode_1 tex_opcode_2 tex_opcode_3 tex_opcode_4)
                            (imp (or (eq tex_geom '.2dms') (eq tex_geom '.a2dms'))
                             (eq tex_ctype '.s32')))

               tex_opcode ::= tex_opcode_1 | tex_opcode_2 | tex_opcode_3 | tex_opcode_4"""

    data3 = """
/*@(CONSTRAINT cvt_opcode_1 (imp (not (eq cvt_1_atype '.f32')) (eq ftz_clause ''))) */

cvt_dty_f ::= float | f16
cvt_aty_f ::= float | f16
cvt_opcode_2 ::= 'cvt' irnd (ftz)? (sat)? cvt_dty_f cvt_aty_f

@(CONSTRAINT cvt_opcode_2 (eq cvt_dty_f cvt_aty_f))

/* TODO: fix this so float to integer conversions are not possible */
cvt_dty_i ::= unsigned | signed | u8 | s8
"""

    out, anno = parse_annotated_grammar(data2)
    print(out, [str(s) for s in anno])

    out2, anno2 = parse_annotated_grammar(data3)
    print(out2, [str(s) for s in anno2])
    
