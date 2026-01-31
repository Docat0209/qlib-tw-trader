"""
模型訓練服務 - IC 增量選擇法
"""

import json
import random
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src.repositories.factor import FactorRepository
from src.repositories.models import Factor, TrainingRun
from src.repositories.training import TrainingRepository
from src.shared.constants import TRAIN_DAYS, VALID_DAYS

TZ_TAIPEI = ZoneInfo("Asia/Taipei")

# 模型檔案目錄
MODELS_DIR = Path("data/models")


@dataclass
class FactorEvalResult:
    """因子評估結果"""

    factor_id: int
    factor_name: str
    ic_value: float
    selected: bool


@dataclass
class TrainingResult:
    """訓練結果"""

    run_id: int
    model_name: str
    model_ic: float
    icir: float | None
    selected_factor_ids: list[int]
    all_results: list[FactorEvalResult]


class ModelTrainer:
    """模型訓練器 - IC 增量選擇法"""

    def __init__(self, qlib_data_dir: Path | str):
        self.data_dir = Path(qlib_data_dir)
        self._qlib_initialized = False
        self._last_ic_std: float | None = None  # 用於 ICIR 計算

    def _init_qlib(self) -> None:
        """初始化 qlib"""
        if self._qlib_initialized:
            return

        try:
            import qlib
            from qlib.config import REG_CN

            qlib.init(
                provider_uri=str(self.data_dir),
                region=REG_CN,
            )
            self._qlib_initialized = True
        except ImportError:
            raise RuntimeError("qlib is not installed. Please run: pip install pyqlib")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize qlib: {e}")

    def _get_instruments(self) -> list[str]:
        """從 qlib instruments 目錄讀取股票清單"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

        if instruments_file.exists():
            with open(instruments_file) as f:
                return [line.strip().split()[0] for line in f if line.strip()]

        # 備選：從 features 目錄取得
        features_dir = self.data_dir / "features"
        if features_dir.exists():
            return [d.name for d in features_dir.iterdir() if d.is_dir()]

        return []

    def get_data_date_range(self) -> tuple[date | None, date | None]:
        """取得 qlib 資料的日期範圍"""
        instruments_file = self.data_dir / "instruments" / "all.txt"

        if not instruments_file.exists():
            return None, None

        min_start = None
        max_end = None

        with open(instruments_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        start = date.fromisoformat(parts[1])
                        end = date.fromisoformat(parts[2])
                        if min_start is None or start < min_start:
                            min_start = start
                        if max_end is None or end > max_end:
                            max_end = end
                    except ValueError:
                        continue

        return min_start, max_end

    def train(
        self,
        session: Session,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> TrainingResult:
        """
        執行 IC 增量選擇訓練

        Args:
            session: 資料庫 Session
            train_start: 訓練開始日期
            train_end: 訓練結束日期
            valid_start: 驗證開始日期
            valid_end: 驗證結束日期
            on_progress: 進度回調 (progress: 0-100, message: str)

        Returns:
            TrainingResult
        """
        factor_repo = FactorRepository(session)
        training_repo = TrainingRepository(session)

        # 取得啟用的因子
        enabled_factors = factor_repo.get_all(enabled=True)
        if not enabled_factors:
            raise ValueError("No enabled factors found")

        candidate_ids = [f.id for f in enabled_factors]

        # 生成模型名稱
        model_name = f"{train_start.strftime('%Y-%m')}~{train_end.strftime('%Y-%m')}"

        # 創建訓練記錄
        run = training_repo.create_run(
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        run.name = model_name
        run.candidate_factor_ids = json.dumps(candidate_ids)
        run.status = "running"
        session.commit()

        if on_progress:
            on_progress(0.0, "Initializing training...")

        try:
            # 執行 IC 增量選擇
            selected_ids, all_results = self._incremental_ic_selection(
                factors=enabled_factors,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                on_progress=on_progress,
            )

            # 計算最終模型 IC
            model_ic = self._calculate_model_ic(
                selected_ids=[f.id for f in enabled_factors if f.id in selected_ids],
                factors=enabled_factors,
                valid_start=valid_start,
                valid_end=valid_end,
            )

            # 計算 ICIR
            icir = self._calculate_icir(model_ic, len(selected_ids))

            # 保存因子結果
            for result in all_results:
                training_repo.add_factor_result(
                    run_id=run.id,
                    factor_id=result.factor_id,
                    ic_value=result.ic_value,
                    selected=result.selected,
                )

            # 完成訓練
            run.selected_factor_ids = json.dumps(list(selected_ids))
            training_repo.complete_run(
                run_id=run.id,
                model_ic=model_ic,
                icir=icir,
                factor_count=len(selected_ids),
            )

            # 保存模型檔案
            self._save_model_files(
                model_name=model_name,
                selected_factors=[f for f in enabled_factors if f.id in selected_ids],
                config={
                    "train_start": train_start.isoformat(),
                    "train_end": train_end.isoformat(),
                    "valid_start": valid_start.isoformat(),
                    "valid_end": valid_end.isoformat(),
                    "model_ic": model_ic,
                    "icir": icir,
                },
            )

            if on_progress:
                on_progress(95.0, "Calculating final metrics...")

            # ... 後續處理 ...

            if on_progress:
                on_progress(100.0, "Training completed")

            return TrainingResult(
                run_id=run.id,
                model_name=model_name,
                model_ic=model_ic,
                icir=icir,
                selected_factor_ids=list(selected_ids),
                all_results=all_results,
            )

        except Exception as e:
            # 標記訓練失敗
            run.status = "failed"
            run.completed_at = datetime.now(TZ_TAIPEI)
            session.commit()
            raise e

    def _incremental_ic_selection(
        self,
        factors: list[Factor],
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> tuple[set[int], list[FactorEvalResult]]:
        """
        IC 增量選擇法

        Returns:
            (selected_factor_ids, all_results)
        """
        # 隨機打亂因子順序
        shuffled_factors = factors.copy()
        random.shuffle(shuffled_factors)

        selected_ids: set[int] = set()
        all_results: list[FactorEvalResult] = []
        current_ic = 0.0

        total = len(shuffled_factors)
        for i, factor in enumerate(shuffled_factors):
            # 進度：5% 初始化 + 90% 因子挑選（每個因子佔 90/total %）
            if on_progress:
                progress = round(5.0 + (i / total) * 90.0, 1)
                on_progress(progress, f"Evaluating factor ({i+1}/{total}): {factor.name}")

            # 計算加入此因子後的 IC
            test_ids = selected_ids | {factor.id}
            new_ic = self._calculate_model_ic(
                selected_ids=list(test_ids),
                factors=factors,
                valid_start=valid_start,
                valid_end=valid_end,
            )

            # 記錄因子的 IC 貢獻
            ic_contribution = new_ic - current_ic

            # 嚴格大於才入選
            if new_ic > current_ic:
                selected_ids.add(factor.id)
                current_ic = new_ic
                selected = True
            else:
                selected = False

            all_results.append(
                FactorEvalResult(
                    factor_id=factor.id,
                    factor_name=factor.name,
                    ic_value=ic_contribution,
                    selected=selected,
                )
            )

            # 評估完成後更新進度
            if on_progress:
                progress = round(5.0 + ((i + 1) / total) * 90.0, 1)
                status = "selected" if selected else "skipped"
                on_progress(progress, f"Factor {factor.name}: {status} (IC: {ic_contribution:.4f})")

        return selected_ids, all_results

    def _calculate_model_ic(
        self,
        selected_ids: list[int],
        factors: list[Factor],
        valid_start: date,
        valid_end: date,
    ) -> float:
        """
        計算模型整體 IC

        IC = 因子值與未來收益率的皮爾遜相關係數
        使用 qlib 計算因子值，然後與未來收益率計算相關性
        """
        if not selected_ids:
            return 0.0

        self._init_qlib()

        from qlib.data import D

        # 1. 取得股票池
        instruments = self._get_instruments()
        if not instruments:
            raise ValueError("No instruments found in qlib data directory")

        # 2. 建立因子表達式列表
        selected_factors = [f for f in factors if f.id in selected_ids]
        if not selected_factors:
            return 0.0

        fields = [f.expression for f in selected_factors]

        # 3. 定義未來收益率 (qlib 預設: T+1 到 T+2 的 1 日報酬)
        label_expr = "Ref($close, -2) / Ref($close, -1) - 1"
        all_fields = fields + [label_expr]

        # 4. 讀取資料
        try:
            df = D.features(
                instruments=instruments,
                fields=all_fields,
                start_time=valid_start.strftime("%Y-%m-%d"),
                end_time=valid_end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load qlib data: {e}")

        if df.empty:
            return 0.0

        # 5. 計算組合因子分數 (標準化後等權平均)
        factor_cols = df.columns[:-1]
        label_col = df.columns[-1]

        # 每日截面標準化
        def zscore_by_date(group: pd.DataFrame) -> pd.DataFrame:
            return (group - group.mean()) / (group.std() + 1e-8)

        factor_zscore = df[factor_cols].groupby(level="datetime", group_keys=False).apply(
            zscore_by_date
        )
        composite_score = factor_zscore.mean(axis=1)

        # 6. 計算每日截面 IC
        combined = pd.DataFrame({
            "score": composite_score,
            "label": df[label_col]
        }).dropna()

        if combined.empty:
            return 0.0

        def calc_corr(group: pd.DataFrame) -> float:
            if len(group) < 2:
                return np.nan
            return group["score"].corr(group["label"])

        daily_ic = combined.groupby(level="datetime", group_keys=False).apply(calc_corr)

        # 7. 計算並保存 IC 標準差（用於 ICIR 計算）
        self._last_ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else None

        # 8. 返回平均 IC
        mean_ic = daily_ic.mean()
        return float(mean_ic) if not np.isnan(mean_ic) else 0.0

    def _calculate_icir(self, ic: float, factor_count: int) -> float | None:
        """計算 IC Information Ratio (ICIR = IC / std(IC))"""
        if factor_count == 0 or ic == 0:
            return None

        # 使用 _calculate_model_ic 計算的 IC 標準差
        ic_std = self._last_ic_std
        if ic_std is None or ic_std == 0:
            return None

        return ic / ic_std

    def _save_model_files(
        self,
        model_name: str,
        selected_factors: list[Factor],
        config: dict,
    ) -> None:
        """保存模型檔案"""
        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_path = model_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 保存因子
        factors_data = [
            {
                "id": f.id,
                "name": f.name,
                "expression": f.expression,
            }
            for f in selected_factors
        ]
        factors_path = model_dir / "factors.json"
        with open(factors_path, "w", encoding="utf-8") as f:
            json.dump(factors_data, f, indent=2, ensure_ascii=False)


def run_training(
    session: Session,
    qlib_data_dir: Path | str,
    train_end: date | None = None,
    on_progress: Callable[[int, str], None] | None = None,
) -> TrainingResult:
    """
    執行訓練的便利函數

    Args:
        session: 資料庫 Session
        qlib_data_dir: qlib 資料目錄
        train_end: 訓練結束日期（預設：今日 - 驗證期天數）
        on_progress: 進度回調
    """
    from datetime import timedelta

    today = date.today()

    if train_end is None:
        train_end = today - timedelta(days=VALID_DAYS)

    train_start = train_end - timedelta(days=TRAIN_DAYS)
    valid_start = train_end + timedelta(days=1)
    valid_end = today

    trainer = ModelTrainer(qlib_data_dir)
    return trainer.train(
        session=session,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        on_progress=on_progress,
    )
