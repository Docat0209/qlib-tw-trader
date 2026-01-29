"""
自訂例外與錯誤處理
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """應用程式基礎例外"""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code


class NotFoundError(AppException):
    """資源未找到"""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            code="NOT_FOUND",
            message=f"找不到 {resource}: {identifier}",
            status_code=404,
        )


class InvalidDateRangeError(AppException):
    """無效的日期範圍"""

    def __init__(self):
        super().__init__(
            code="INVALID_DATE_RANGE",
            message="結束日期必須大於等於開始日期",
            status_code=400,
        )


class ValidationError(AppException):
    """驗證錯誤"""

    def __init__(self, message: str):
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            status_code=400,
        )


def register_exception_handlers(app: FastAPI):
    """註冊例外處理器"""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )
