from typing import Any


class AppException(Exception):
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, detail: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.context = context or {}


class NotFoundException(AppException):
    status_code = 404
    error_code = "NOT_FOUND"


class ConflictException(AppException):
    status_code = 409
    error_code = "CONFLICT"


class UnauthorizedException(AppException):
    status_code = 401
    error_code = "UNAUTHORIZED"


class ForbiddenException(AppException):
    status_code = 403
    error_code = "FORBIDDEN"


class ValidationException(AppException):
    status_code = 422
    error_code = "VALIDATION_ERROR"


class GraphException(AppException):
    status_code = 500
    error_code = "GRAPH_ERROR"


class RecommendationException(AppException):
    status_code = 500
    error_code = "RECOMMENDATION_ERROR"
