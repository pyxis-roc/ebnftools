#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 University of Rochester
#
# SPDX-License-Identifier: MIT

import argparse
from ebnftools.ebnfast import EBNFParser, generate_dot, generate_graph
from ebnftools.ebnfgrammar import EBNFAnnotatedGrammar

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Visualize an EBNF AST as a graph")
    p.add_argument("grammar", help="EBNF grammar")
    p.add_argument("output_prefix", help="Output DOT file prefix")
    args = p.parse_args()

    gr = EBNFAnnotatedGrammar()
    with open(args.grammar, "r") as f:
        grraw = f.read()
        gr.parse(grraw)

    for r in gr.rules:
        el = gr.ast_graph(r, True)
        g = generate_dot(el)

        fn = args.output_prefix + "_" + r.lhs.value + ".dot"
        with open(fn, "w") as f:
            print(fn)
            for l in g:
                print(l, file=f)

