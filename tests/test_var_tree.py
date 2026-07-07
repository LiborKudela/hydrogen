"""Unit tests for the `var_tree` UI tree builder (`host._build_var_tree`).

Pure function over a list of dotted variable names -- no host process needed.
Locks down the dropdown-tree contract a UI relies on: explicit ``leaf`` flags,
unique ``path`` keys, subtree ``count`` badges, leaves-first natural ordering,
the stripped common root, and the leaf-that-is-also-a-prefix edge case.
"""

from __future__ import annotations

from hydrogen.service.host import _build_var_tree


def _by_name(node):
    return {c["name"]: c for c in node["children"]}


def test_basic_grouping_paths_leaves_and_count():
    tree = _build_var_tree(["sys.src.y", "sys.lag.y", "sys.lag.u"])

    # The common root ("sys") is stripped from display paths.
    assert tree["name"] == "" and tree["path"] == "" and tree["leaf"] is False
    assert tree["count"] == 3  # three selectable variables

    top = _by_name(tree)
    assert set(top) == {"lag", "src"}
    assert top["lag"]["leaf"] is False and top["lag"]["path"] == "lag"
    assert top["lag"]["count"] == 2
    assert top["src"]["count"] == 1

    lag = _by_name(top["lag"])
    assert lag["u"]["leaf"] is True
    assert lag["u"]["path"] == "lag.u"
    assert lag["u"]["full"] == "sys.lag.u"
    assert isinstance(lag["u"]["index"], int)
    # Pure leaves are not expandable: no children key.
    assert "children" not in lag["u"]


def test_paths_are_unique_across_tree():
    names = ["a.b.c", "a.b.d", "a.e", "f.g"]
    tree = _build_var_tree(names)
    seen = []

    def walk(node):
        seen.append(node["path"])
        for c in node.get("children", []):
            walk(c)

    walk(tree)
    # Root path "" plus every distinct sub-path, no duplicates.
    assert len(seen) == len(set(seen))
    assert "" in seen and "a.b.c" in seen and "a.b" in seen


def test_indices_match_input_order():
    names = ["m.alpha", "m.beta", "m.gamma"]
    tree = _build_var_tree(names)
    leaves = {c["full"]: c["index"] for c in tree["children"]}
    assert leaves == {"m.alpha": 0, "m.beta": 1, "m.gamma": 2}


def test_leaves_before_groups_and_natural_sort():
    # Mix of a group ("grp") and bare leaves, plus numbered siblings.
    names = [
        "r.zeta",          # leaf
        "r.alpha",         # leaf
        "r.grp.x",         # under a group
        "r.seg10.v",
        "r.seg2.v",
    ]
    tree = _build_var_tree(names)
    order = [c["name"] for c in tree["children"]]

    # Bare leaves (alpha, zeta) -- a level's own, outermost variables -- come
    # before groups (grp, seg2, seg10)...
    groups = [c["name"] for c in tree["children"] if not c["leaf"]]
    leaves = [c["name"] for c in tree["children"] if c["leaf"]]
    assert order == leaves + groups
    # ...leaves alphabetical, groups natural-sorted (seg2 before seg10).
    assert leaves == ["alpha", "zeta"]
    assert groups == ["grp", "seg2", "seg10"]


def test_leaf_that_is_also_a_prefix():
    # "x.tank" is recorded *and* is a prefix of "x.tank.level".
    tree = _build_var_tree(["x.tank", "x.tank.level"])
    tank = tree["children"][0]
    assert tank["name"] == "tank"
    # It is selectable (a leaf) yet also expandable (has children).
    assert tank["leaf"] is True
    assert tank["full"] == "x.tank"
    assert "children" in tank and tank["children"][0]["name"] == "level"
    # count includes the node itself (1) plus its one descendant.
    assert tank["count"] == 2


def test_empty_input():
    tree = _build_var_tree([])
    assert tree == {"name": "", "path": "", "leaf": False,
                    "children": [], "count": 0}
