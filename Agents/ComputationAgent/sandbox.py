"""Secure sandbox for executing deterministic computation snippets."""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, Iterable, Mapping, Optional


_ALLOWED_BUILTINS: Mapping[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "round",
        "range",
        "enumerate",
        "sorted",
        "map",
        "filter",
        "list",
        "dict",
        "set",
        "tuple",
        "zip",
        "float",
        "int",
        "str",
        "bool",
        "any",
        "all",
    )
}

_ALLOWED_MODULES: Mapping[str, Any] = {
    "math": math,
    "statistics": statistics,
    "datetime": datetime,
    "date": date,
    "timedelta": timedelta,
    "Decimal": Decimal,
    "Fraction": Fraction,
}

_FORBIDDEN_NAMES: Iterable[str] = (
    "__import__",
    "eval",
    "exec",
    "open",
    "compile",
    "globals",
    "locals",
    "vars",
    "help",
    "input",
)


class SandboxViolation(RuntimeError):
    """Raised when submitted Python violates sandbox guardrails."""


class _SafetyChecker(ast.NodeVisitor):
    """AST visitor enforcing a conservative subset of Python."""

    _allowed_nodes = {
        ast.Module,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Expr,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Subscript,
        ast.Slice,
        ast.Index,
        ast.If,
        ast.IfExp,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Return,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Constant,
        ast.Lambda,
        ast.With,
        ast.alias,
    }

    _forbidden_nodes = {
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Delete,
        ast.Try,
        ast.Raise,
        ast.Global,
        ast.Nonlocal,
        ast.Yield,
        ast.YieldFrom,
    }

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        if type(node) in self._forbidden_nodes:
            raise SandboxViolation(f"Node type {type(node).__name__} is not permitted in sandbox")
        if type(node) not in self._allowed_nodes:
            raise SandboxViolation(f"Node type {type(node).__name__} is not supported in sandbox")
        return super().visit(node)

    def visit_Call(self, node: ast.Call) -> Any:  # type: ignore[override]
        self._validate_callable(node.func)
        return super().visit_Call(node)

    def visit_Name(self, node: ast.Name) -> Any:  # type: ignore[override]
        if node.id in _FORBIDDEN_NAMES:
            raise SandboxViolation(f"Usage of name '{node.id}' is not permitted in sandbox")
        if node.id.startswith("__"):
            raise SandboxViolation("Magic dunder names are not permitted")
        return super().visit_Name(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:  # type: ignore[override]
        if node.attr.startswith("__"):
            raise SandboxViolation("Attribute access to magic methods is not permitted")
        return super().visit_Attribute(node)

    def _validate_callable(self, call: ast.expr) -> None:
        if isinstance(call, ast.Name):
            if call.id not in _ALLOWED_BUILTINS and call.id not in _ALLOWED_MODULES:
                raise SandboxViolation(f"Callable '{call.id}' is not allowed in sandbox")
        elif isinstance(call, ast.Attribute):
            root = self._resolve_root_name(call)
            if root not in _ALLOWED_MODULES:
                raise SandboxViolation(
                    f"Attribute callable '{ast.unparse(call)}' is not exposed in sandbox"
                )
        else:
            raise SandboxViolation("Indirect callables are not permitted")

    @staticmethod
    def _resolve_root_name(node: ast.Attribute) -> str:
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value  # type: ignore[assignment]
        if isinstance(current, ast.Name):
            return current.id
        raise SandboxViolation("Unsupported callable expression in sandbox")


@dataclass(slots=True)
class SandboxExecution:
    """Output of a sandboxed computation."""

    result: Any
    stdout: str
    locals: Dict[str, Any]

    def as_tabular(self) -> Optional[Dict[str, Any]]:
        table = self.locals.get("table")
        if table is None:
            return None

        if isinstance(table, dict):
            rows = table.get("rows")
            columns = table.get("columns")
            if isinstance(rows, list) and isinstance(columns, list):
                return {
                    "columns": [str(col) for col in columns],
                    "rows": [dict(row) if isinstance(row, Mapping) else row for row in rows],
                    "row_count": len(rows),
                }
        if isinstance(table, list):
            if not table:
                return {"columns": [], "rows": [], "row_count": 0}
            if isinstance(table[0], Mapping):
                columns = list(table[0].keys())
                return {
                    "columns": [str(col) for col in columns],
                    "rows": [dict(row) for row in table],
                    "row_count": len(table),
                }
        return None


class SafeComputationSandbox:
    """Executes Python snippets with conservative guardrails."""

    def execute(self, code: str, *, context: Optional[Dict[str, Any]] = None) -> SandboxExecution:
        trimmed = code.strip()
        if not trimmed:
            raise SandboxViolation("No code supplied for computation")

        module = ast.parse(trimmed)
        checker = _SafetyChecker()
        checker.visit(module)

        globals_env: Dict[str, Any] = {"__builtins__": dict(_ALLOWED_BUILTINS)}
        globals_env.update(_ALLOWED_MODULES)
        globals_env["context"] = dict(context or {})

        locals_env: Dict[str, Any] = {}
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            exec(compile(module, "<sandbox>", "exec"), globals_env, locals_env)

        if "result" not in locals_env:
            raise SandboxViolation("Computed code must assign a 'result' variable")

        return SandboxExecution(result=locals_env["result"], stdout=buffer.getvalue(), locals=locals_env)


__all__ = [
    "SandboxExecution",
    "SandboxViolation",
    "SafeComputationSandbox",
]


