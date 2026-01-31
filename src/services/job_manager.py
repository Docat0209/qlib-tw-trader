"""
JobManager - 非同步任務管理器

管理任務執行和 WebSocket 進度推送
"""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import WebSocket

from src.repositories.database import get_session
from src.repositories.job import JobRepository
from src.shared.constants import TZ_TAIPEI


class ConnectionManager:
    """WebSocket 連線管理器"""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """接受 WebSocket 連線"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """移除 WebSocket 連線"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """廣播訊息給所有連線"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


# 全域連線管理器
manager = ConnectionManager()


class JobManager:
    """任務管理器"""

    def __init__(self):
        self._running_jobs: dict[str, asyncio.Task] = {}

    async def create_job(
        self,
        job_type: str,
        task_fn: Callable,
        message: str | None = None,
        **kwargs,
    ) -> str:
        """
        建立並執行任務

        Args:
            job_type: 任務類型 (train/sync/backtest)
            task_fn: 任務函數（必須是 async def）
            message: 初始訊息
            **kwargs: 傳給 task_fn 的參數

        Returns:
            job_id
        """
        session = get_session()
        repo = JobRepository(session)

        # 建立任務記錄
        job = repo.create(job_type=job_type, message=message)
        job_id = job.id
        session.close()

        # 廣播任務建立
        await manager.broadcast({
            "type": "job_created",
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "message": message,
        })

        # 建立 async task
        task = asyncio.create_task(
            self._run_job(job_id, job_type, task_fn, **kwargs)
        )
        self._running_jobs[job_id] = task

        return job_id

    async def _run_job(
        self,
        job_id: str,
        job_type: str,
        task_fn: Callable,
        **kwargs,
    ):
        """執行任務"""
        session = get_session()
        repo = JobRepository(session)

        try:
            # 更新為執行中
            repo.update_status(job_id, "running", progress=0)
            await manager.broadcast({
                "type": "job_progress",
                "job_id": job_id,
                "job_type": job_type,
                "status": "running",
                "progress": 0,
            })

            # 執行任務，傳入 progress callback（支援浮點數）
            async def progress_callback(progress: float, message: str | None = None):
                repo.update_status(job_id, "running", progress=progress, message=message)
                await manager.broadcast({
                    "type": "job_progress",
                    "job_id": job_id,
                    "job_type": job_type,
                    "status": "running",
                    "progress": progress,
                    "message": message,
                })

            result = await task_fn(progress_callback=progress_callback, **kwargs)

            # 完成
            result_json = json.dumps(result) if result else None
            repo.complete(job_id, result=result_json, success=True)
            await manager.broadcast({
                "type": "job_completed",
                "job_id": job_id,
                "job_type": job_type,
                "status": "completed",
                "progress": 100,
                "result": result,
            })

        except Exception as e:
            # 失敗
            repo.complete(job_id, result=str(e), success=False)
            await manager.broadcast({
                "type": "job_failed",
                "job_id": job_id,
                "job_type": job_type,
                "status": "failed",
                "error": str(e),
            })

        finally:
            session.close()
            if job_id in self._running_jobs:
                del self._running_jobs[job_id]

    def get_running_jobs(self) -> list[str]:
        """取得執行中的任務 ID"""
        return list(self._running_jobs.keys())

    async def cancel_job(self, job_id: str) -> bool:
        """
        取消執行中的任務

        Returns:
            True if cancelled, False if not found
        """
        task = self._running_jobs.get(job_id)
        if task:
            task.cancel()
            del self._running_jobs[job_id]

            # 更新資料庫狀態
            session = get_session()
            repo = JobRepository(session)
            repo.complete(job_id, result="Cancelled by user", success=False)
            session.close()

            # 廣播取消事件
            await manager.broadcast({
                "type": "job_cancelled",
                "job_id": job_id,
                "status": "cancelled",
            })
            return True

        return False


# 全域任務管理器
job_manager = JobManager()


async def broadcast_data_updated(
    entity: str,
    action: str,
    entity_id: str | int | None = None,
) -> None:
    """
    廣播資料更新事件

    Args:
        entity: 資料類型 ("factors" | "models" | "datasets" | "backtests")
        action: 操作類型 ("create" | "update" | "delete")
        entity_id: 資料 ID
    """
    await manager.broadcast({
        "type": "data_updated",
        "entity": entity,
        "action": action,
        "entity_id": str(entity_id) if entity_id else None,
        "timestamp": datetime.now(TZ_TAIPEI).isoformat(),
    })
