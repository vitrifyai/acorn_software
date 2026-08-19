"""
Static audit of the controller-mixin split.

MainWindow's methods live in five files now. The failure that split introduces is
quiet: a method lands in a module whose imports do not cover every name it uses.
The module still imports, the test suite still collects, and the NameError only
arrives when a user walks that particular path — which is how a missing
`import numpy as np` in movie.py survived long enough to reach a screenshot.

This resolves every name each function loads against its own scope, its enclosing
scopes, its module, and builtins. Used by the test suite; also runnable by hand:

    python -m acorn.gui._split_audit
"""
from __future__ import annotations

import ast
import builtins
from typing import Iterable, Iterator

MIXIN_MODULES = (
    "acorn.gui.main_window",
    "acorn.gui.sam_controller",
    "acorn.gui.detector_controller",
    "acorn.gui.export_controller",
    "acorn.gui.movie",
    "acorn.gui.threads",
)

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _child_scopes(node: ast.AST) -> Iterator[ast.AST]:
    """Nested functions and lambdas directly under *node*, not their own children."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPES):
            yield child
        else:
            yield from _child_scopes(child)


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Every node belonging to this scope, stopping at nested function boundaries.

    Nested scopes are analysed separately, with this scope's bindings as context,
    so descending into them here would attribute their parameters to the parent.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPES):
            continue
        yield child
        yield from _own_nodes(child)


def _bindings(node: ast.AST, include_args: bool = True) -> set[str]:
    """Names bound in this scope: parameters, assignments, imports, except, loops."""
    bound: set[str] = set()
    args = getattr(node, "args", None)
    if include_args and args is not None:
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            bound.add(a.arg)
        for a in (args.vararg, args.kwarg):
            if a is not None:
                bound.add(a.arg)

    for child in _own_nodes(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            bound.add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            bound.update(child.names)
    # a nested def/lambda binds its own name in the enclosing scope
    for scope in _child_scopes(node):
        name = getattr(scope, "name", None)
        if name:
            bound.add(name)
    return bound


def _walk_scopes(node: ast.AST, enclosing: set[str], out: dict[str, set[str]],
                 path: str = "") -> None:
    bound = _bindings(node) | enclosing
    loaded = {
        n.id for n in _own_nodes(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    missing = loaded - bound
    if missing and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        out[path or node.name] = missing
    for scope in _child_scopes(node):
        child_name = getattr(scope, "name", "<lambda>")
        _walk_scopes(scope, bound, out, f"{path or getattr(node, 'name', '')}.{child_name}")


# Names Python puts in every module namespace.
_MODULE_DUNDERS = frozenset({
    "__name__", "__file__", "__doc__", "__package__", "__loader__", "__spec__",
    "__builtins__", "__debug__", "__annotations__",
})


def undefined_names(module_or_path) -> dict[str, set[str]]:
    """
    Names each function loads that nothing in scope can supply. Empty means sound.

    Scope is read from the source, never from the imported module's ``vars()``: a
    module object keeps whatever a previous import left in it, so checking against
    it would happily "verify" a file whose import line has been deleted. Accepts a
    module or a path so a file can be checked without importing it.
    """
    path = getattr(module_or_path, "__file__", module_or_path)
    source = open(path, encoding="utf-8").read()
    tree = ast.parse(source)
    module_scope = (
        set(dir(builtins)) | _MODULE_DUNDERS | _bindings(tree, include_args=False)
    )
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
           for n in ast.walk(tree)):
        return {}      # a star-import makes the module scope unknowable statically

    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _walk_scopes(item, module_scope, out, f"{node.name}.{item.name}")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_scopes(node, module_scope, out, node.name)
    return out


def audit(module_names: Iterable[str] = MIXIN_MODULES) -> dict[str, dict[str, set[str]]]:
    """Run undefined_names over each module; returns only modules with problems."""
    import importlib

    return {
        name: found
        for name in module_names
        if (found := undefined_names(importlib.import_module(name)))
    }


if __name__ == "__main__":       # pragma: no cover - convenience for manual runs
    import json
    report = audit()
    print(json.dumps({m: {f: sorted(v) for f, v in fs.items()}
                      for m, fs in report.items()}, indent=2) if report else "clean")
