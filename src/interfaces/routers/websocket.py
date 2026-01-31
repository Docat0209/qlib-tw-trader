"""
WebSocket API - 即時任務進度推送
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from src.interfaces.dependencies import get_db
from src.repositories.job import JobRepository
from src.services.job_manager import job_manager, manager

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 連線端點

    接收訊息格式:
    - {"type": "ping"} -> 回應 {"type": "pong"}
    - {"type": "subscribe", "job_id": "xxx"} -> 訂閱特定任務

    推送訊息格式:
    - {"type": "job_created", "job_id": "xxx", ...}
    - {"type": "job_progress", "job_id": "xxx", "progress": 50, ...}
    - {"type": "job_completed", "job_id": "xxx", "result": {...}}
    - {"type": "job_failed", "job_id": "xxx", "error": "..."}
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@router.get("/jobs")
async def list_jobs(
    session: Session = Depends(get_db),
    limit: int = 20,
):
    """取得任務列表"""
    repo = JobRepository(session)
    jobs = repo.get_recent(limit)
    return {
        "items": [
            {
                "id": j.id,
                "job_type": j.job_type,
                "status": j.status,
                "progress": j.progress,
                "message": j.message,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ],
        "total": len(jobs),
    }


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    session: Session = Depends(get_db),
):
    """取得任務詳情"""
    repo = JobRepository(session)
    job = repo.get(job_id)
    if not job:
        return {"error": "Job not found"}

    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    session: Session = Depends(get_db),
):
    """取消/刪除任務"""
    repo = JobRepository(session)
    job = repo.get(job_id)

    if not job:
        return {"error": "Job not found", "status": "error"}

    # 如果是 running 或 queued，嘗試取消正在執行的任務
    if job.status in ("running", "queued"):
        await job_manager.cancel_job(job_id)  # 嘗試取消（可能不存在）

    # 無論如何都從資料庫刪除
    session.delete(job)
    session.commit()

    # 廣播取消事件
    await manager.broadcast({
        "type": "job_cancelled",
        "job_id": job_id,
        "status": "deleted",
    })

    return {"status": "deleted", "id": job_id}
