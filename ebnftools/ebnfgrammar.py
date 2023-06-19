# SPDX-FileCopyrightText: 2020,2021,2023 University of Rochester
#
# SPDX-License-Identifier: MIT

#
# ebnfgrammar.py
#
# Expose grammars as objects, to simplify the API

from . import ebnfast
from . import ebnfanno
from collections import OrderedDict, namedtuple

Treepos = namedtuple('Treepos', 'node current original previous')

class EBNFGrammar(object):
    @property
    def rules(self):
        for l in self.raw:
            if isinstance(l, ebnfast.Rule):
                yield l

    @property
    def ruledict(self):
        # we don't cache this since we have no good way of checking if raw has changed...
        if not hasattr(self, '_rd') or self._rd is None:
            self._rd = dict([(r.lhs.value, r.rhs) for r in self.rules])

        return self._rd

    @property
    def raw(self):
        for r in self._raw:
            yield r

    def _set_raw(self, new_raw):
        self._raw = new_raw
        self._rd = None

    def name_objects(self, objects, namer, *namer_args):
        for obj in objects:
            if hasattr(obj, '_name'):
                oldname = obj._name
            else:
                oldname = None

            obj._name = namer(obj, oldname, *namer_args)

    def get_treepos(self, rule):
        def add_to_treepos(n, tp):
            cur = None # result of compute_treepos
            orig = None # created during conversion
            prev = None # result of previous compute_treepos

            if hasattr(n, '_treepos'):
                cur = n._treepos

            if  hasattr(n, '_from_treepos'):
                orig = n._from_treepos

            if hasattr(n, '_old_treepos'):
                prev = n._old_treepos

            tp.append(Treepos(current = cur, original = orig, previous = prev, node = n))


        # use a list, so we don't make assumptions about duplicates
        # which can occur, esp. for _from_treepos
        treepos = []

        ebnfast.visit_rule_dict(self.ruledict, rule.rhs, add_to_treepos,
                                [treepos], descend_symbols = False)

        return treepos

    def compute_treepos(self, rule, use_ns = True):
        path_to_objs = OrderedDict()
        ebnfast.compute_treepos(rule.rhs, path_to_objs, ns = rule.lhs.value if use_ns else None)

        return path_to_objs

    def ast_graph(self, root, expand_symbols = False):
        """Generate a graph (i.e. edge list) starting at root, which is
           usually a Rule. If expand_symbols is True, then the tree
           will recurse into symbols.

           The edge list can be passed to an output routine like `generate_dot`.

        """

        edgelist = {}
        rd = None
        if expand_symbols: rd = self.ruledict

        ebnfast.visualize_ast(root, edgelist, rd)

        return edgelist

    def parse(self, grammar: str, preserve_comments = False):
        if preserve_comments:
            raise NotImplementedError(f"Ability to preserve comments is not implemented yet!")

        parser = ebnfast.EBNFParser()
        p = parser.parse(grammar)

        self._set_raw(p)

    def from_rules(self, rules):
        self._set_raw(rules)

class EBNFAnnotatedGrammar(EBNFGrammar):
    @property
    def rules(self):
        for l in self._ebnf: # anno will never contain rules
            if isinstance(l, ebnfast.Rule):
                yield l

    @property
    def anno(self):
        for a in self._anno:
            yield a

    def filter_anno(self, only: set):
        for a in self.anno:
            name = a.value[0].value
            if name in only:
                yield a

    @property
    def raw(self):
        grit = iter(self._ebnf)
        annoit = iter(self._anno)

        gr_done = False
        anno_done = False

        try:
            gr = next(grit)
        except StopIteration:
            gr_done = True

        try:
            anno = next(annoit)
        except StopIteration:
            anno_done = True

        while not (gr_done and anno_done):
            try:
                while not gr_done and (anno_done or gr.coord.order < anno.coord.order):
                    yield gr
                    gr = next(grit)
            except StopIteration:
                gr_done = True

            try:
                while not anno_done and (gr_done or anno.coord.order < gr.coord.order):
                    yield anno
                    anno = next(annoit)
            except StopIteration:
                anno_done = True

    def _set_raw(self, new_raw):
        super(EBNFAnnotatedGrammar, self)._set_raw(new_raw)
        self._ebnf = self._raw

    def parse(self, grammar: str, preserve_comments = False):
        gr, anno = ebnfanno.parse_annotated_grammar(grammar)

        self._anno = anno # this is parsed
        super(EBNFAnnotatedGrammar, self).parse('\n'.join(gr), preserve_comments)

    def from_rules(self, rules, anno = []):
        self._anno = anno # this is parsed

        # order is a bad design... remove it
        # make annotations follow the rule they reference

        last_order = None
        for r in rules:
            if not hasattr(r, 'coord'):
                assert False, r

            #print(r, r.coord.order)
            if last_order is not None:
                if not (last_order < r.coord.order):
                    assert len(last_order) != r.coord.order, f"{r} {r.coord.order} {last_order}"

            last_order = r.coord.order

        super(EBNFAnnotatedGrammar, self).from_rules(rules)
