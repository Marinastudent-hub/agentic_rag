"""Computation tools: calculator and analyze_data."""
import ast
import math
import operator as op
import re

import numpy as np
import pandas as pd
from langchain_core.tools import tool

# ---------------- Calculator (safe: no eval) ----------------
_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
}
_UNARY_OPS = {ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "pow": pow,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Only numbers are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large.")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("Unsupported expression.")


def _fmt_number(x: float) -> str:
    if isinstance(x, float) and math.isfinite(x):
        x = round(x, 6)
        if x == int(x) and abs(x) < 1e15:
            x = int(x)
    return str(x)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression exactly. ALWAYS use this for arithmetic instead
    of calculating in your head (sums, differences, percentages, ratios, growth).
    Supports + - * / ** % ( ), sqrt(), abs(), round(), min(), max(), log().
    Write numbers WITHOUT thousands separators and without currency symbols,
    e.g. '(4.2 - 3.5) / 3.5 * 100'. A percent sign works: '4200000 * 15%'.
    Args:
      expression: the math expression as plain text."""
    expr = (expression.strip().replace("$", "").replace("^", "**")
            .replace("×", "*").replace("÷", "/"))
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    try:
        result = _eval(ast.parse(expr, mode="eval"))
    except Exception as e:
        return (f"Calculation error: {e} "
                "Rewrite the expression using only numbers and + - * / ( ).")
    return f"{expression.strip()} = {_fmt_number(result)}"


# ---------------- Data analysis ----------------
def _n(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _parse_numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text or "")]


@tool
def analyze_data(numbers: str, operation: str = "summary", labels: str = "") -> str:
    """Analyze a list of numbers that you found in the documents.
    Args:
      numbers: comma-separated numbers WITHOUT thousands separators, e.g. '3.5, 4.2'.
      operation: one of
        'summary' (count, sum, mean, median, min, max, std),
        'growth'  (percent change between consecutive values and overall),
        'trend'   (is the series increasing or decreasing, and how fast),
        'rank'    (sort from highest to lowest).
      labels: optional comma-separated labels in the same order, e.g. 'Q2, Q3'.
    For a single arithmetic calculation, use calculator instead."""
    nums = _parse_numbers(numbers)
    if not nums:
        return "No numbers found. Provide comma-separated numbers, e.g. '3.5, 4.2'."

    label_list = [l.strip() for l in (labels or "").split(",") if l.strip()]
    if len(label_list) != len(nums):
        label_list = [f"#{i + 1}" for i in range(len(nums))]

    operation = (operation or "summary").strip().lower()
    s = pd.Series(nums)

    if operation == "summary":
        lines = [
            f"Count: {len(s)}",
            f"Sum: {_n(s.sum())}",
            f"Mean: {_n(s.mean())}",
            f"Median: {_n(s.median())}",
            f"Min: {_n(s.min())} ({label_list[int(s.idxmin())]})",
            f"Max: {_n(s.max())} ({label_list[int(s.idxmax())]})",
        ]
        if len(s) > 1:
            lines.append(f"Std deviation: {_n(s.std())}")
        return "\n".join(lines)

    if operation == "growth":
        if len(nums) < 2:
            return "Growth needs at least 2 numbers."
        lines = []
        for i in range(1, len(nums)):
            prev, cur = nums[i - 1], nums[i]
            if prev == 0:
                change = "n/a (previous value is 0)"
            else:
                change = f"{(cur - prev) / abs(prev) * 100:+.2f}%"
            lines.append(f"{label_list[i - 1]} -> {label_list[i]}: {change} "
                         f"({_n(prev)} -> {_n(cur)})")
        if len(nums) > 2 and nums[0] != 0:
            total = (nums[-1] - nums[0]) / abs(nums[0]) * 100
            lines.append(f"Overall {label_list[0]} -> {label_list[-1]}: {total:+.2f}%")
        return "\n".join(lines)

    if operation == "trend":
        if len(nums) < 2:
            return "Trend needs at least 2 numbers."
        slope = float(np.polyfit(range(len(nums)), nums, 1)[0])
        mean = abs(float(s.mean())) or 1.0
        if abs(slope / mean) < 0.01:
            direction = "flat (no clear trend)"
        else:
            direction = "increasing" if slope > 0 else "decreasing"
        return (f"Trend: {direction}\n"
                f"Average change per period: {slope:+.4f}\n"
                f"First: {_n(nums[0])} ({label_list[0]}), "
                f"Last: {_n(nums[-1])} ({label_list[-1]})")

    if operation == "rank":
        order = sorted(range(len(nums)), key=lambda i: nums[i], reverse=True)
        return "\n".join(f"{r}. {label_list[i]}: {_n(nums[i])}"
                         for r, i in enumerate(order, start=1))

    return "Unknown operation. Use one of: summary, growth, trend, rank."