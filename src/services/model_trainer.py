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

    def train(
        self,
        session: Session,
        train_start: date,
        train_end: date,
        valid_start: date,
        valid_end: date,
        on_progress: Callable[[int, str], None] | None = None,
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
            on_progress(5, "Training started")

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
                on_progress(100, "Training completed")

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
        on_progress: Callable[[int, str], None] | None = None,
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
            if on_progress:
                progress = 10 + int((i / total) * 80)  # 10-90%
                on_progress(progress, f"Evaluating factor: {factor.name}")

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

        TODO: 實作真正的 qlib IC 計算
        目前使用模擬值進行測試
        """
        if not selected_ids:
            return 0.0

        # TODO: 使用 qlib 計算真正的 IC
        # 暫時使用模擬邏輯（基於因子數量）
        base_ic = 0.02
        factor_bonus = len(selected_ids) * 0.005
        noise = random.uniform(-0.01, 0.01)
        return min(base_ic + factor_bonus + noise, 0.15)

    def _calculate_icir(self, ic: float, factor_count: int) -> float | None:
        """計算 IC Information Ratio"""
        if factor_count == 0:
            return None
        # ICIR = IC / std(IC)，這裡簡化計算
        return ic / 0.05 if ic > 0 else None

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
