"""
Job Repository - 非同步任務資料存取
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.repositories.models import Job


class JobRepository:
    """非同步任務 Repository"""

    def __init__(self, session: Session):
        self._session = session

    def create(self, job_type: str, message: str | None = None) -> Job:
        """建立新任務"""
        job = Job(
            id=str(uuid4()),
            job_type=job_type,
            status="queued",
            progress=0,
            message=message,
        )
        self._session.add(job)
        self._session.commit()
        self._session.refresh(job)
        return job

    def get(self, job_id: str) -> Job | None:
        """取得任務"""
        stmt = select(Job).where(Job.id == job_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_active(self) -> list[Job]:
        """取得進行中的任務"""
        stmt = select(Job).where(Job.status.in_(["queued", "running"]))
        return list(self._session.execute(stmt).scalars().all())

    def get_recent(self, limit: int = 20) -> list[Job]:
        """取得最近的任務"""
        stmt = select(Job).order_by(Job.started_at.desc()).limit(limit)
        return list(self._session.execute(stmt).scalars().all())

    def update_status(
        self,
        job_id: str,
        status: str,
        progress: float | None = None,
        message: str | None = None,
    ) -> Job | None:
        """更新任務狀態"""
        job = self.get(job_id)
        if not job:
            return None

        job.status = status
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message

        self._session.commit()
        self._session.refresh(job)
        return job

    def complete(
        self,
        job_id: str,
        result: str | None = None,
        success: bool = True,
    ) -> Job | None:
        """完成任務"""
        job = self.get(job_id)
        if not job:
            return None

        job.status = "completed" if success else "failed"
        job.progress = 100 if success else job.progress
        job.result = result
        job.completed_at = datetime.now()

        self._session.commit()
        self._session.refresh(job)
        return job
