#!/usr/bin/env python3

from ebnftools import ebnfast
from ebnftools.convert.bnf import EBNF2BNF
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert EBNF2BNF")
    p.add_argument("ebnf")
    args = p.parse_args()

    pr = ebnfast.EBNFParser()
    with open(args.ebnf, "r") as f:
        gr = f.read()

    rules = pr.parse(gr)
    xf = EBNF2BNF()
    bnf = xf.visit_RuleList(rules)
    for r in bnf:
        print(r)
