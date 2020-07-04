#
# ebnfgrammar.py
#
# Expose grammars as objects, to simplify the API

from . import ebnfast
from . import ebnfanno

class EBNFGrammar(object):
    @property
    def rules(self):
        for l in self.raw:
            if isinstance(l, ebnfast.Rule):
                yield l

    @property
    def ruledict(self):
        # we don't cache this since we have no good way of checking if raw has changed...
        return dict([(r.rhs.value, r.rhs) for r in self.rules])

    @property
    def raw(self):
        for r in self._raw:
            yield r

    def parse(self, grammar: str, preserve_comments = False):
        if preserve_comments:
            raise NotImplementedError(f"Ability to preserve comments is not implemented yet!")

        parser = ebnfast.EBNFParser()
        p = parser.parse(grammar)

        self._raw = p

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
        # for now yield all grammar first then anno
        # TODO: better interleaving

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

    def parse(self, grammar: str, preserve_comments = False):
        gr, anno = ebnfanno.parse_annotated_grammar(grammar)

        self._anno = anno # this is parsed

        super(EBNFAnnotatedGrammar, self).parse('\n'.join(gr), preserve_comments)

        # super sets _raw which is NOT used by EBNFAnnotatedGrammar
        self._ebnf = self._raw
