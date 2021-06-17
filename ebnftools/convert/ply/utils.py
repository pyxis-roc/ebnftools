#!/usr/bin/env python3

def make_concat_list(ct, sel=None):
    """Convert the concrete parse tree of a bnf_concat_* rule into a list"""

    if ct is None:
        return

    x = ct
    if sel is None: sel = range(0,len(ct.args)-1)
    while x is not None:
        for andx in sel:
            yield x.args[andx]
        x = x.args[-1]

def dfs_token_list_rec(ct):
    """Construct a flat list of all the string tokens encountered in a concrete parse tree."""

    out = []
    for x in ct.args:
        if isinstance(x, str):
            out.append(x)
        elif not (x is None):
            out.extend(dfs_token_list_rec(x))

    return out

dfs_token_list = dfs_token_list_rec

def vis_parse_tree(root, out = None):
    is_root = False

    if out is None:
        out = ["digraph {"]
        is_root = True

    if not hasattr(root, 'args'):
        print(f"ERROR: {root} has no 'args' attribute. Visualization will be garbled")
        return ""

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
            try:
                root._dotid = f"node_{len(out)}"
            except AttributeError:
                print(f"ERROR: Could not set rootid. Visualization will be garbled")
                return ""

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
