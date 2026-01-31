"""
因子表達式驗證器

驗證 qlib 因子表達式的語法正確性。
"""

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """驗證結果"""

    valid: bool
    error: str | None = None
    fields_used: list[str] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FactorValidator:
    """因子表達式驗證器"""

    # 可用欄位（從 QlibExporter.DAILY_FIELDS 取得）
    VALID_FIELDS = {
        # OHLCV
        "open",
        "high",
        "low",
        "close",
        "volume",
        # 還原股價
        "adj_close",
        # 估值
        "pe_ratio",
        "pb_ratio",
        "dividend_yield",
        # 三大法人
        "foreign_buy",
        "foreign_sell",
        "trust_buy",
        "trust_sell",
        "dealer_buy",
        "dealer_sell",
        # 融資融券
        "margin_buy",
        "margin_sell",
        "margin_balance",
        "short_buy",
        "short_sell",
        "short_balance",
        # 外資持股
        "total_shares",
        "foreign_shares",
        "foreign_ratio",
        "foreign_remaining_shares",
        "foreign_remaining_ratio",
        "foreign_upper_limit_ratio",
        "chinese_upper_limit_ratio",
        # 借券
        "lending_volume",
        # 月營收
        "revenue",
    }

    # qlib 支援的運算符
    VALID_OPERATORS = {
        # 時間序列
        "Ref",
        "Delta",
        "Slope",
        "Rsquare",
        "Resi",
        # 滾動視窗
        "Mean",
        "Sum",
        "Std",
        "Var",
        "Max",
        "Min",
        "Rank",
        "Quantile",
        "Count",
        "Med",
        "Mad",
        "Skew",
        "Kurt",
        "Corr",
        "Cov",
        # 指數平滑
        "EMA",
        "WMA",
        # 元素運算
        "Abs",
        "Sign",
        "Log",
        "Power",
        # 比較運算
        "Greater",
        "Less",
        "Gt",
        "Ge",
        "Lt",
        "Le",
        "Eq",
        "Ne",
        # 邏輯運算
        "And",
        "Or",
        "Not",
        "If",
        "Mask",
        # 配對運算
        "Add",
        "Sub",
        "Mul",
        "Div",
    }

    def validate(self, expression: str) -> ValidationResult:
        """
        驗證因子表達式

        Args:
            expression: qlib 因子表達式

        Returns:
            ValidationResult
        """
        if not expression or not expression.strip():
            return ValidationResult(valid=False, error="Expression is empty")

        expression = expression.strip()
        warnings = []

        # 1. 檢查括號配對
        paren_error = self._check_parentheses(expression)
        if paren_error:
            return ValidationResult(valid=False, error=paren_error)

        # 2. 提取並驗證欄位
        fields_used = self._extract_fields(expression)
        unknown_fields = [f for f in fields_used if f not in self.VALID_FIELDS]
        if unknown_fields:
            return ValidationResult(
                valid=False,
                error=f"Unknown field(s): ${', $'.join(unknown_fields)}",
            )

        # 3. 提取並驗證運算符
        operators_used = self._extract_operators(expression)
        unknown_ops = [op for op in operators_used if op not in self.VALID_OPERATORS]
        if unknown_ops:
            return ValidationResult(
                valid=False,
                error=f"Unknown operator(s): {', '.join(unknown_ops)}",
            )

        # 4. 檢查常見錯誤
        if "/0" in expression.replace(" ", ""):
            warnings.append("Potential division by zero detected")

        # 5. 檢查是否至少使用了一個欄位
        if not fields_used:
            return ValidationResult(
                valid=False,
                error="Expression must reference at least one field (e.g., $close)",
            )

        return ValidationResult(
            valid=True,
            fields_used=fields_used,
            operators_used=operators_used,
            warnings=warnings,
        )

    def _check_parentheses(self, expression: str) -> str | None:
        """檢查括號配對"""
        count = 0
        for i, char in enumerate(expression):
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
                if count < 0:
                    return f"Unmatched closing parenthesis at position {i}"

        if count > 0:
            return f"Missing {count} closing parenthesis(es)"

        return None

    def _extract_fields(self, expression: str) -> list[str]:
        """提取表達式中使用的欄位"""
        # 匹配 $fieldname 格式
        pattern = r"\$([a-zA-Z_][a-zA-Z0-9_]*)"
        matches = re.findall(pattern, expression)
        return list(set(matches))

    def _extract_operators(self, expression: str) -> list[str]:
        """提取表達式中使用的運算符"""
        # 匹配 OperatorName( 格式
        pattern = r"([A-Z][a-zA-Z]*)\s*\("
        matches = re.findall(pattern, expression)
        return list(set(matches))

    def get_available_fields(self) -> list[str]:
        """取得所有可用欄位"""
        return sorted(self.VALID_FIELDS)

    def get_available_operators(self) -> list[str]:
        """取得所有可用運算符"""
        return sorted(self.VALID_OPERATORS)
