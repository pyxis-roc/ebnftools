#!/usr/bin/env python3
#
# ebnfast.py
#
# An AST for EBNF grammars, modeled on the W3 XML Formal Grammar
# (https://www.w3.org/TR/xml/#sec-notation)


import re

class Coord(object):
    def __init__(self, order, linestart = None, lineend = None, colstart = None, colend = None):
        self.order = (order,) if isinstance(order, int) else order # monotonically increasing based on some opaque criteria
        self.line = (linestart, lineend)
        self.col = (colstart, colend)

    def __str__(self):
        return f"{self.line}:{self.col}:{self.order}"

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

    def __repr__(self):
        return f"Symbol({repr(self.value)})"

    def graph(self):
        return (str(self), [])

class Char(Expression):
    precedence = 0
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return f"#x{self.value:x}"

    def __repr__(self):
        return f"Char({repr(self.v)})"

    def graph(self):
        return (str(self), [])

class String(Expression):
    precedence = 0
    def __init__(self, v):
        self.value = v

    def __str__(self):
        return "'" + self.value + "'"

    def __repr__(self):
        return f"String({repr(self.value)})"

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

    def __repr__(self):
        return f"CharClass({repr(str(self))})"

    def graph(self):
        return (str(self), [])

class Optional(Expression):
    precedence = 0

    def __init__(self, expr):
        self.expr = expr

    @property
    def children(self):
        return [self.expr]

    def __str__(self):
        return f"{self.paren(self.expr)}?"

    def __repr__(self):
        return f"Optional({repr(self.expr)})"

    def graph(self):
        return ("?", self.children)

class BinOp(Expression):
    op = None

    def __init__(self, a, b):
        self.expr = [a, b]

    @property
    def children(self):
        return self.expr

    def __str__(self):
        x1 = self.paren(self.expr[0])
        x2 = self.paren(self.expr[1])
        o = self.op
        #return f"{x1}{self.op}{x2}"

        if self.op == ' | ':
            # only do this if we were top-level
            #if len(x1) > 50:
            #    o = '\n        | '
            pass

        return f"{x1}{o}{x2}"

    def __repr__(self):
        return f"{self.__classname__}({repr(self.expr[0])}, {repr(self.expr[1])})"

    def graph(self):
        return (self.op.strip(), self.children)

class Sequence(BinOp):
    precedence = 0
    op = ' '

    def __repr__(self):
        return f"Sequence({repr(self.expr[0])}, {repr(self.expr[1])})"

class Alternation(BinOp):
    precedence = 2
    op = ' | '

    def __repr__(self):
        return f"Alternation({repr(self.expr[0])}, {repr(self.expr[1])})"


class Subtraction(BinOp):
    precedence = 3 #???
    op = ' - '

    def __repr__(self):
        return f"Subtraction({repr(self.expr[0])}, {repr(self.expr[1])})"

# handles both one or more, zero or more
class Concat(Expression):
    precedence = 0

    def __init__(self, expr, minimum: int = 0):
        self.expr = expr
        self.minimum = minimum

    @property
    def children(self):
        return [self.expr]

    def __str__(self):
        suffix = '*' if self.minimum == 0 else '+'
        #print("STR", self.expr)
        return f"{self.paren(self.expr)}{suffix}"

    def __repr__(self):
        return f"Concat({repr(self.expr)}, {self.minimum})"

    def graph(self):
        return ('*' if self.minimum == 0 else '+', self.children)

class Rule(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    @property
    def children(self):
        return [self.lhs, self.rhs]

    def __str__(self):
        return f"{self.lhs} ::= {self.rhs}"

    def __repr__(self):
        return f"Rule({repr(self.lhs)}, {repr(self.rhs)})"

    def graph(self):
        return ('::=', self.children)


def visit_rule_dict(grammar, node, vfn, vfn_args = [], descend_symbols = True, _visited = None):
    if _visited is None:
        _visited = set()

    vfn(node, *vfn_args)

    if isinstance(node, Symbol):
        if descend_symbols and node.value not in _visited:
            _visited.add(node.value)
            visit_rule_dict(grammar, grammar[node.value], vfn, vfn_args, descend_symbols, _visited)
    else:
        for c in node.children:
            # print("\tvisiting", node, c, node.children)
            visit_rule_dict(grammar, c, vfn, vfn_args, _visited)

def visit_rules(rules, start, vfn, vfn_args = []):
    grammar = dict([(r.lhs.value, r.rhs) for r in rules])
    if start == '*':
        # visit all the rules
        for r in rules:
            n = grammar[r.lhs.value]
            vfn(r, *vfn_args) # visit rule
            vfn(r.lhs, *vfn_args) # visit LHS symbol
            visit_rule_dict(grammar, n, vfn, vfn_args)
    else:
        start_node = grammar[start]
        visit_rule_dict(grammar, start_node, vfn, vfn_args)

def visualize_ast(root, edgelist, rule_dict = None):
    key = id(root)

    if key in edgelist:
        return

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

def unfold_associative(root, cls):
    """Converts cls(a, cls(b, cls(c, d))) to [a, b, c, d]"""
    out = []

    if not isinstance(root.expr[0], cls):
        out.append(root.expr[0])
    else:
        out.extend(unfold_associative(root.expr[0], cls))

    if not isinstance(root.expr[1], cls):
        out.append(root.expr[1])
    else:
        out.extend(unfold_associative(root.expr[1], cls))

    return out

def compute_treepos(obj, path_to_objs, path_prefix = (),
                    parent = None, ns = None, node_index = 1, _DEBUG = False):
    """Compute treepos values for each item in obj (which, when invoked, is rule.rhs).

       Sequence: A B C D gets number 1 2 3 4
       Alternation: A | B | C | D gets 1 for the whole alternation, then (1, 1) for A, ..., (1, 4) for D
    """

    treepos = (*path_prefix, node_index)

    if ns:
        nstreepos = (ns, treepos)
    else:
        nstreepos = treepos

    if hasattr(obj, '_treepos'):
        obj._old_treepos = obj._treepos

    if _DEBUG:
        print(obj, "\t", parent, "\t*", parent._treepos if parent is not None and hasattr(parent, '_treepos') else "", "*\t", treepos, node_index)

    if isinstance(obj, Symbol):
        obj._treepos = nstreepos
        path_to_objs[obj._treepos] = obj
        return 1
    elif isinstance(obj, String):
        obj._treepos = nstreepos
        path_to_objs[obj._treepos] = obj
        return 1
    elif isinstance(obj, Subtraction):
        obj._treepos = nstreepos
        path_to_objs[obj._treepos] = obj

        # don't support internal naming for subtraction
        return 1
    elif isinstance(obj, CharClass):
        obj._treepos = nstreepos
        path_to_objs[obj._treepos] = obj

        return 1
    elif isinstance(obj, (Optional, Concat)):
        obj._treepos = nstreepos
        path_to_objs[obj._treepos] = obj

        compute_treepos(obj.expr, path_to_objs, treepos, obj, ns, 1)
        return 1
    elif isinstance(obj, Alternation):
        obj._treepos = nstreepos
        path_to_objs[obj._treepos] = obj
        seq = unfold_associative(obj, Alternation)
        if _DEBUG: print(seq)

        for node_index, s in enumerate(seq, 1):
            if isinstance(s, Sequence):
                # provide a handle for entire sequence
                if ns:
                    s._treepos = (ns, (*treepos, node_index))
                else:
                    s._treepos = (*treepos, node_index)

                path_to_objs[s._treepos] = s

                compute_treepos(s, path_to_objs, (*treepos, node_index), obj, ns, 1, _DEBUG)
            else:
                compute_treepos(s, path_to_objs, treepos, obj, ns, node_index, _DEBUG)

        return 1
    elif isinstance(obj, Sequence):
        #obj._treepos = treepos # no way to refer to sequence objects?
        #path_to_objs[obj._treepos] = obj

        consumed = compute_treepos(obj.expr[0], path_to_objs, tuple(treepos[:-1]), obj, ns, node_index, _DEBUG)
        consumed += compute_treepos(obj.expr[1], path_to_objs, tuple(treepos[:-1]),  obj, ns, node_index + consumed, _DEBUG)
        return consumed
    else:
        raise NotImplementedError(f"Don't support {obj}")

#TODO: comments, [ wfc: ] [ vc: ] occur at the end

# this is a bit more verbose than ASTNodeTransformer, but that's okay
class EBNFTransformer(object):
    def visit_Rule(self, node):
        node.lhs = self.visit(node.lhs)
        node.rhs = self.visit(node.rhs)

        return node

    def visit_Concat(self, node):
        node.expr = self.visit(node.expr)

        return node

    # todo: possibly useful if one of expr is zero to return the other
    # (except for subtract)

    def visit_BinOp(self, node):
        node.expr = [self.visit(x) for x in node.expr]
        return node

    def visit_Optional(self, node):
        self.expr = self.visit(node.expr)
        return node

    def visit_CharClass(self, node):
        return node

    def visit_String(self, node):
        return node

    def visit_Char(self, node):
        return node

    def visit_Symbol(self, node):
        return node

    def visit_Subtraction(self, node):
        return self.visit_BinOp(node)

    def visit_Alternation(self, node):
        return self.visit_BinOp(node)

    def visit_Sequence(self, node):
        return self.visit_BinOp(node)

    def visit(self, node):
        if isinstance(node, Rule):
            return self.visit_Rule(node)
        elif isinstance(node, Concat):
            return self.visit_Concat(node)
        elif isinstance(node, BinOp):
            if isinstance(node, Subtraction):
                return self.visit_Subtraction(node)
            elif isinstance(node, Alternation):
                return self.visit_Alternation(node)
            elif isinstance(node, Sequence):
                return self.visit_Sequence(node)
            else:
                raise NotImplementedError(f"Unknown BinOp node {node}")
        elif isinstance(node, CharClass):
            return self.visit_CharClass(node)
        elif isinstance(node, Symbol):
            return self.visit_Symbol(node)
        elif isinstance(node, Char):
            return self.visit_Char(node)
        elif isinstance(node, String):
            return self.visit_String(node)
        elif isinstance(node, Optional):
            return self.visit_Optional(node)
        elif isinstance(node, Concat):
            return self.visit_Concat(node)
        else:
            raise NotImplementedError(f"Unimplemented visit for node {type(node)}")

    # this is not in the AST
    def visit_RuleList(self, rule_list):
        new_list = []
        for r in rule_list:
            nr = self.visit(r)
            if nr is None: # delete rule
                continue

            if isinstance(nr, list): # replace with multiple rules
                new_list.extend(nr)
            else: # replace with another (or the same) rule
                new_list.append(nr)

        return new_list

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
            sline = token_stream.coord[0]

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
                rule = Rule(Symbol(lhs), rhs)
                rule.coord = Coord(sline) # TODO...
                out.append(rule)
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

    g = generate_graph(x[0], generate_dot, dict([(r.lhs.value, r.rhs) for r in x]))
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

def test_xformer():
    p = EBNFParser()

    rules = p.parse('''
Char	   ::=   	#x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]

S	   ::=   	(#x20 | #x9 | #xD | #xA)+

Comment	   ::=   	'<!--' ((Char - '-') | ('-' (Char - '-')))* '-->'

content	   ::=   	CharData? ((element | Reference | CDSect | PI | Comment) CharData?)*
''')

    xf = EBNFTransformer()
    rl = xf.visit_RuleList(rules)
    print(rules, rl)

def test_visitor_infinite_loop():
    p = EBNFParser()
    rules = p.parse("""
wsp ::= [#x20#x9]
bnf_concat_instruction ::= wsp | wsp bnf_concat_instruction
instruction ::= 'add' bnf_concat_instruction args ';'
bnf_concat_args ::= [^;] | [^;] bnf_concat_args
args ::= bnf_concat_args
""")

    def _visit(n):
        print(n)

    # shouldn't raise RecursionError
    visit_rules(rules, '*', _visit)
