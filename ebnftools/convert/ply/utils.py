#!/usr/bin/env python3

def make_concat_list(ct):
    """Convert the concrete parse tree of a bnf_concat_* rule into a list"""

    x = ct
    while x is not None:
        yield x.args[0]
        x = x.args[1]

def vis_parse_tree(root, out = None):
    is_root = False

    if out is None:
        out = ["digraph {"]
        is_root = True

    if root is None:
        nid = f"none_{len(out)}"
        out.append(f'{nid} [label=""]')
    elif isinstance(root, str):
        nid = f"str_{len(out)}"
        out.append(f'{nid} [label="{root}"]')
    else:
        cids = []
        for c in root.args:
            cids.append(vis_parse_tree(c, out))

        if not hasattr(root, '_dotid'):
            root._dotid = f"node_{len(out)}"
            n = root.__class__.__name__
            out.append(f'{root._dotid} [label="{n}"];')

            for c in cids:
                out.append(f'{root._dotid} -> {c};')

        nid = root._dotid

    if is_root:
        out.append("}")
        return out
    else:
        return nid

def visit_abstract(root):
    if not hasattr(root, 'abstract'):
        return root

    if not hasattr(root, 'args'):
        return root

    if len(root.args) == 0:
        if hasattr(root, 'abstract'):
            return root.abstract()
        else:
            return root
    else:
        out = []
        for x in root.args:
            if x is not None:
                out.append(visit_abstract(x))
            else:
                out.append(x)

        root.args = out
        return root.abstract()
