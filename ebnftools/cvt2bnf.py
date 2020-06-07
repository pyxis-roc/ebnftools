#!/usr/bin/env python3

from ebnftools import ebnfast
from ebnftools.ebnfgen import isfinite, count, generate2, flatten
from ebnftools.convert.bnf import EBNF2BNF, get_strings, get_charclass, LiteralRewriter, CharClassRewriter
from ebnftools.convert.tokens import TokenRegistry, TknLiteral, TknRegExp, TknCharClass
import argparse
import copy
import pathlib
import sys

def sanity_check(orig, new, paranoid = True):
    orig_as_dict = dict([(r.lhs.value, r.rhs) for r in orig])
    new_as_dict = dict([(r.lhs.value, r.rhs) for r in new])

    for k in orig_as_dict:
        if k not in new_as_dict:
            print(f"ERROR: {k} is *not* in bnf", file=sys.stderr)
        else:
            if isfinite(orig_as_dict, orig_as_dict[k]) and isfinite(new_as_dict, new_as_dict[k]):
                # check for weak-equivalence by counting
                # ideally we would check the strings in the language as well

                co = count(orig_as_dict, orig_as_dict[k])
                cn = count(new_as_dict, new_as_dict[k])
                if co != cn:
                    print(f"ERROR: Count for rule {k}: EBNF count {co} != BNF count {cn}", file=sys.stderr)
                else:
                    print(f"INFO: {k} is okay [{cn} == {co}]", file=sys.stderr)

                if paranoid:
                    s1 = set(generate2(orig_as_dict, orig_as_dict[k]))
                    s2 = set(generate2(new_as_dict, new_as_dict[k]))

                    if s1 != s2:
                        print(f"ERROR: Generated strings for {k} don't match: {s1 - s2} and {s2 - s1}", file=sys.stderr)
                    else:
                        print(f"INFO: Generated strings for {k} match", file=sys.stderr)
            else:
                print(f"ERROR: Can't check {k} for weak equivalence, contains infinite productions", file=sys.stderr)

def _add_token(treg, prefix, value, k = 0):
    try_token_name = prefix + f"_{k}"

    while try_token_name in treg.tokens:
        try_token_name = prefix + f"_{k}"
        k += 1

    treg.add(try_token_name, value)

    return k

def tokenize_cc_literals(bnf, treg):
    all_cc = get_charclass(bnf)

    k = 0
    for s in all_cc:
        sk = TknCharClass(s[1:-1])

        if sk.key() not in treg.v2n:
            k = _add_token(treg, "TOKEN_CC", sk, k)

    return treg

def tokenize_string_literals(bnf, treg):
    all_strings = get_strings(bnf)

    k = 0
    for s in all_strings:
        if s == '': continue # empty strings are handled differently

        sk = TknLiteral(s)
        if sk.key() not in treg.v2n:
            k = _add_token(treg, "TOKEN_STR", sk, k)

    return treg

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert EBNF2BNF")
    p.add_argument("ebnf")
    p.add_argument("tokens", nargs="?", help="File containing tokens (used as input/output)")
    p.add_argument("--check", action="store_true",
                   help="Check that resulting BNF is `equivalent' (when possible)")

    args = p.parse_args()

    pr = ebnfast.EBNFParser()
    with open(args.ebnf, "r") as f:
        gr = f.read()

    rules = pr.parse(gr)
    orig_rules = copy.deepcopy(rules)
    xf = EBNF2BNF()
    bnf = xf.visit_RuleList(rules)

    if args.tokens:
        tokens = {}
        tkns = pathlib.Path(args.tokens)
        treg = TokenRegistry(args.tokens)
        if tkns.exists(): treg.read()

        treg = tokenize_string_literals(bnf, treg)
        treg = tokenize_cc_literals(bnf, treg)
        treg.write()

        rwt = LiteralRewriter()
        rwt.rewrite(bnf, treg.v2n)

        ccrwt = CharClassRewriter()
        ccrwt.rewrite(bnf, treg.v2n)

        for t, s in treg.n2v.items():
            if not isinstance(s, TknRegExp):
                # note that s are subclasses of String/CharClass
                print(ebnfast.Rule(ebnfast.Symbol(t), s))
            else:
                # this is usually not a problem since the token will
                # be in the lexer anyway, but it would be nice to see
                # if the regexp can be converted to EBNF
                print(f"Regular expressions literals in EBNF are not supported, omitting token '{t} ::= {s}'", file=sys.stderr)

    for r in bnf:
        print(r)

    if args.check:
        sanity_check(orig_rules, bnf, True)
