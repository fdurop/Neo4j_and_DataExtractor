"""Normalization, validation, and deduplication for extracted formulas."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


class FormulaValidator:
    """Apply deterministic checks before a formula enters formal KG data."""

    _UNCERTAIN_MARKERS = ("?", "？", "无法识别", "不确定", "uncertain")

    def normalize(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(formula)
        latex = str(item.get("latex") or item.get("content") or "").strip()
        latex = re.sub(r"^```(?:latex)?\s*|\s*```$", "", latex, flags=re.IGNORECASE)
        latex = re.sub(r"^\s*LaTeX\s*[:：]\s*", "", latex, flags=re.IGNORECASE)
        if latex.startswith("$$") and latex.endswith("$$") and len(latex) >= 4:
            latex = latex[2:-2].strip()
        elif latex.startswith("$") and latex.endswith("$") and len(latex) >= 2:
            latex = latex[1:-1].strip()
        elif latex.startswith(r"\[") and latex.endswith(r"\]"):
            latex = latex[2:-2].strip()

        item["latex"] = latex
        item["plain_text"] = str(item.get("plain_text") or item.get("content") or latex).strip()
        item["name"] = str(item.get("name") or "").strip()
        item["meaning"] = str(item.get("meaning") or item.get("description") or "").strip()
        item["description"] = item["meaning"]
        item["symbols"] = item.get("symbols") if isinstance(item.get("symbols"), list) else []
        item["conditions"] = item.get("conditions") if isinstance(item.get("conditions"), list) else []
        item["uncertain_symbols"] = (
            item.get("uncertain_symbols") if isinstance(item.get("uncertain_symbols"), list) else []
        )
        item.setdefault("source_type", item.get("extraction_method") or "unknown")
        item.setdefault("extraction_method", item["source_type"])
        return item

    def validate(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        item = self.normalize(formula)
        latex = item["latex"]
        issues: List[str] = []

        if item.get("review_required") and item.get("review_reason"):
            issues.append(str(item["review_reason"]))

        if not latex and "empty_latex" not in issues:
            issues.append("empty_latex")
        if latex and self._is_atomic_symbol(latex):
            issues.append("not_substantive_formula")
        if latex and self._is_isolated_fragment(latex):
            issues.append("isolated_formula_fragment")
        if latex and self._has_possible_case_collision(latex):
            issues.append("possible_symbol_case_confusion")
        if latex and not self._balanced(latex):
            issues.append("unbalanced_delimiters")
        if latex and self._has_incomplete_command(latex):
            issues.append("incomplete_latex_command")
        if item["uncertain_symbols"] or any(marker in latex.lower() for marker in self._UNCERTAIN_MARKERS):
            issues.append("uncertain_symbols")
        if item.get("is_complete_formula") is False:
            issues.append("model_reports_incomplete_formula")
        if len(latex) > 1500:
            issues.append("latex_too_long")

        source_type = str(item.get("source_type") or "")
        default_confidence = 0.94 if "latex" in source_type else 0.84
        if "simple" in source_type or "equation" in source_type:
            default_confidence = 0.78
        if "image" in source_type or item.get("backend") == "api":
            default_confidence = 0.86
        try:
            confidence = float(item.get("confidence", default_confidence))
        except (TypeError, ValueError):
            confidence = default_confidence
        confidence -= 0.28 * len(issues)
        confidence = max(0.0, min(1.0, confidence))

        pass_threshold = 0.85 if "image" in source_type or item.get("backend") == "api" else 0.70
        review_required = bool(issues) or confidence < pass_threshold
        item.update(
            {
                "confidence": round(confidence, 4),
                "validation_issues": issues,
                "validation_status": "review" if review_required else "passed",
                "review_required": review_required,
                "review_reason": issues[0] if issues else ("low_confidence" if review_required else None),
                "review_status": "pending" if review_required else None,
                "indexable": bool(latex) and not review_required,
            }
        )
        return item

    def validate_many(self, formulas: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected: Dict[str, Dict[str, Any]] = {}
        for formula in self._merge_continuations(formulas or []):
            item = self.validate(formula)
            key = self._dedupe_key(item)
            if not key:
                key = f"empty:{len(selected)}"
            previous = selected.get(key)
            if previous is None or item["confidence"] > previous["confidence"]:
                selected[key] = item
        return list(selected.values())

    def _merge_continuations(self, formulas: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Join equation lines beginning with '=' or an implication arrow."""
        merged: List[Dict[str, Any]] = []
        continuation = re.compile(r"^\s*(?:=|\\(?:Rightarrow|Leftarrow|Leftrightarrow|implies)\b)")
        for original in formulas or []:
            current = dict(original)
            latex = str(current.get("latex") or current.get("content") or "").strip()
            same_source = bool(merged) and (
                current.get("source_image") == merged[-1].get("source_image")
                and current.get("image_index") == merged[-1].get("image_index")
            )
            if latex and same_source and continuation.match(latex):
                previous = merged[-1]
                previous_latex = str(previous.get("latex") or previous.get("content") or "").strip()
                previous["latex"] = f"{previous_latex} {latex}".strip()
                previous["content"] = previous["latex"]
                uncertainty = list(previous.get("uncertain_symbols") or [])
                for symbol in current.get("uncertain_symbols") or []:
                    if symbol not in uncertainty:
                        uncertainty.append(symbol)
                previous["uncertain_symbols"] = uncertainty
                try:
                    previous["confidence"] = min(
                        float(previous.get("confidence", 1.0)),
                        float(current.get("confidence", 1.0)),
                    )
                except (TypeError, ValueError):
                    pass
                continue
            merged.append(current)
        return merged

    @staticmethod
    def _dedupe_key(item: Dict[str, Any]) -> str:
        latex = re.sub(r"\s+", "", str(item.get("latex") or ""))
        latex = latex.replace(r"\left", "").replace(r"\right", "")
        return latex.lower()

    @staticmethod
    def _balanced(text: str) -> bool:
        pairs = {"}": "{", ")": "(", "]": "["}
        stack: List[str] = []
        escaped = False
        for char in text:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char in "{([":
                stack.append(char)
            elif char in "})]":
                if not stack or stack.pop() != pairs[char]:
                    return False
        return not stack

    @staticmethod
    def _has_incomplete_command(latex: str) -> bool:
        if re.search(r"\\(?:frac|dfrac|tfrac)\s*(?:$|\{[^{}]*\}\s*$)", latex):
            return True
        if re.search(r"\\(?:sqrt|text|mathrm|mathbf)\s*$", latex):
            return True
        return bool(re.search(r"[_^]\s*$", latex))

    @staticmethod
    def _is_atomic_symbol(latex: str) -> bool:
        """Reject labels such as u_s(t) that are better represented as concepts."""
        substantive_markers = (
            "=", "+", "-", "/", "\\frac", "\\sum", "\\int", "\\lim",
            "\\cdot", "\\times", "\\Rightarrow", "\\Leftrightarrow",
        )
        if any(marker in latex for marker in substantive_markers):
            return False
        stripped = re.sub(r"\\(?:tilde|widetilde|hat|bar|mathbf|mathrm|text)\s*", "", latex)
        stripped = re.sub(r"[{}_^\s]", "", stripped)
        return bool(re.fullmatch(r"[A-Za-zΑ-Ωα-ω]+(?:\([^()]*\))?", stripped))

    @staticmethod
    def _is_isolated_fragment(latex: str) -> bool:
        """Flag short labels or operands that need surrounding visual context."""
        compact = re.sub(r"\s+", "", latex)
        if re.fullmatch(r"\\(?:pm|mp)[+-]?\d+(?:\.\d+)?", compact):
            return True
        relations = ("=", "<", ">", r"\le", r"\ge", r"\approx", r"\sim", r"\propto", r"\Rightarrow")
        substantive = (r"\int", r"\sum", r"\prod", r"\lim", r"\begin")
        if any(marker in compact for marker in relations + substantive):
            return False
        # Expressions such as d/4 or Delta-theta/2 are commonly diagram labels
        # or pieces of prose. Retain them for teacher review instead of deleting.
        return len(compact) <= 32 and bool(re.search(r"[A-Za-zΑ-Ωα-ω\\]", compact))

    @staticmethod
    def _has_possible_case_collision(latex: str) -> bool:
        """Detect self-scaling equations where VLM case confusion is plausible."""
        compact = re.sub(r"\s+", "", latex)
        match = re.fullmatch(
            r"([A-Za-z])=\1(?:\\?cdot|\\?times|[*/])(.+)",
            compact,
        )
        return bool(match)
