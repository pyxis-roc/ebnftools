#!/usr/bin/env python3
#
# tokens.py
#
# Although EBNF does not require a strict separation between tokens
# and non-terminals, some other formats do, and this file provides
# utilities to make it easier to specify tokens in a portable manner.

import re
from ..ebnfast import String, CharClass

class TknLiteral(String):
    def key(self):
        return repr(self.value)

class TknCharClass(CharClass):
    def key(self):
        return str(self)

class TknRegExp(object):
    def __init__(self, re):
        self.value = re # must be a string in /regexp/ format

    def key(self):
        return str(self)

    def __str__(self):
        return "/" + self.value + "/"

class TokenRegistry(object):
    """A token registry is a file that maps token names to patterns. The
       file contains one token per line, formatted as two fields:

         TOKEN_NAME value

       where value is one of:

       'literal': A string literal
       [charclass]: A character class, to encode EBNF charclasses
       /regexp/: A regular expression (which can include charclasses). These are not part of EBNF and
                 are almost always manually specified
    """

    def __init__(self, fn):
        self.fn = fn
        self.tokens = set()
        self.v2n = {}
        self.n2v = {}

    def remove(self, token):
        if token not in self.tokens:
            raise KeyError(f"Token {token} not found")

        value = self.n2v[token]

        assert value.key() in self.v2n, f"Internal inconsistency between n2v and v2n, token {token} value {value} not found in v2n"

        del self.n2v[token]
        del self.v2n[value.key()]
        self.tokens.remove(token)

    def add(self, token, value):
        assert isinstance(value, (TknLiteral, TknCharClass, TknRegExp)), f"Incorrect type {type(value)} for {value}"

        if token in self.tokens:
            raise ValueError(f"Duplicate token {token}")

        if value.key() in self.v2n:
            raise ValueError(f"Duplicate value {value}")

        self.v2n[value.key()] = token
        self.n2v[token] = value
        self.tokens.add(token)

    def read(self):
        v2n = {}
        n2v = {}
        tokens = set()

        with open(self.fn, "r") as f:
            for lno, l in enumerate(f, 1):
                ls = l.strip().split(' ', 1)
                if ls[0] == "#": continue

                if not (ls[1] and  ls[0]):
                    raise ValueError(f"ERROR:{lno}: Line is malformed, empty value ({value}) and/or token name ({name})")

                name, value = ls[0], ls[1]
                if value in v2n:
                    # note: this does not detect semantic equality, just structural
                    raise ValueError(f"ERROR:{lno}: Value {value} is duplicated")

                if name in tokens:
                    raise ValueError(f"ERROR:{lno}: Token {name} is duplicated")

                if value[0] != value[-1]:
                    if value[0] != '[' and value[-1] != ']':
                        raise ValueError(f"ERROR:{lno}: Value {value} is incorrectly specified")

                if value[0] == "'" or value[0] == '"':
                    value = TknLiteral(value[1:-1])
                elif value[0] == "[":
                    value = TknCharClass(value[1:-1])
                elif value[0] == "/":
                    value = TknRegExp(value[1:-1])

                v2n[value.key()] = name
                n2v[name] = value
                tokens.add(name)

        self.v2n = v2n
        self.n2v = n2v
        self.tokens = tokens

    def write(self, filename = None):
        if filename is None: filename = self.fn

        with open(filename, "w") as f:
            for s, t in self.v2n.items():
                print(f"{t} {s}", file=f)
