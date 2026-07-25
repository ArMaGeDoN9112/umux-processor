"""Optional FastAPI adapter for the shared UMUX processing service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import AsyncIterator
import zipfile

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from umux_processor.artifacts import ARTIFACT_FILENAMES
from umux_processor.config import ConfigurationError, load_configuration
from umux_processor.ingestion import IngestionError
from umux_processor.pipeline import PipelineServiceError, run_pipeline_service


LOGGER = logging.getLogger(__name__)
CSV_MEDIA_TYPES = frozenset({"text/csv", "application/csv", "application/vnd.ms-excel"})
CONFIG_MEDIA_TYPES = frozenset({"application/toml", "text/plain"})
EXPECTED_ARTIFACTS = (*ARTIFACT_FILENAMES, "dashboard.html")


@dataclass(frozen=True)
class ApiSettings:
    """Bounded storage settings for the HTTP upload boundary."""

    temp_root: Path
    max_files: int
    max_file_bytes: int
    max_request_bytes: int
    default_config: Path = Path("/app/config/normalization.toml")

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        return cls(
            temp_root=Path(os.environ.get("UMUX_API_TEMP_ROOT", "/tmp/umux-api")),
            max_files=_positive_environment("UMUX_API_MAX_FILES", 10),
            max_file_bytes=_positive_environment("UMUX_API_MAX_FILE_BYTES", 10 * 1024 * 1024),
            max_request_bytes=_positive_environment("UMUX_API_MAX_REQUEST_BYTES", 25 * 1024 * 1024),
        )


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


def _positive_environment(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _sanitize_filename(filename: str | None, index: int) -> str:
    """Return a safe, non-empty basename used only inside a request workspace."""
    basename = Path((filename or "upload.csv").replace("\\", "/")).name
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return f"upload-{index}-{normalized or 'data.csv'}"


def _error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings.from_environment()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        settings.temp_root.mkdir(parents=True, exist_ok=True)
        LOGGER.info("API started")
        try:
            yield
        finally:
            LOGGER.info("API stopped")

    app = FastAPI(title="UMUX Processor API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def enforce_declared_request_limit(request: Request, call_next: object) -> object:
        if request.url.path == "/process" and request.method == "POST":
            content_type = request.headers.get("content-type", "").lower()
            if not content_type.startswith("multipart/form-data"):
                return _error_response(ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be multipart/form-data."))
        length = request.headers.get("content-length")
        if length is not None:
            try:
                if int(length) > settings.max_request_bytes:
                    return _error_response(ApiError(413, "REQUEST_TOO_LARGE", "Request exceeds the configured size limit."))
            except ValueError:
                return _error_response(ApiError(400, "MALFORMED_REQUEST", "Invalid Content-Length header."))
        return await call_next(request)  # type: ignore[operator]

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        return _error_response(error)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
        return _error_response(ApiError(400, "MALFORMED_MULTIPART", "A valid multipart request with at least one files field is required."))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        code = "MALFORMED_MULTIPART" if error.status_code == 400 else "HTTP_ERROR"
        message = "Malformed multipart request." if error.status_code == 400 else "HTTP request could not be processed."
        return _error_response(ApiError(error.status_code, code, message))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, __: Exception) -> JSONResponse:
        LOGGER.error("Unexpected API processing failure")
        return _error_response(ApiError(500, "PROCESSING_FAILED", "The request could not be processed."))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/process", responses={
        400: {"description": "Malformed multipart request"},
        413: {"description": "Configured request limit exceeded"},
        415: {"description": "Unsupported upload media type"},
        422: {"description": "Configuration, CSV schema, or input error"},
    })
    async def process(
        request: Request,
        files: list[UploadFile] = File(...),
        config: UploadFile | None = File(default=None),
    ) -> FileResponse:
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be multipart/form-data.")
        if len(files) > settings.max_files:
            raise ApiError(413, "TOO_MANY_FILES", "Too many uploaded CSV files.")

        workspace = tempfile.TemporaryDirectory(prefix="request-", dir=settings.temp_root)
        try:
            workspace_path = Path(workspace.name)
            total_bytes = 0
            inputs: list[Path] = []
            for index, upload in enumerate(files, start=1):
                if (upload.content_type or "").lower() not in CSV_MEDIA_TYPES:
                    raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Uploaded data files must have a supported CSV media type.")
                target = workspace_path / _sanitize_filename(upload.filename, index)
                total_bytes += await _save_upload(upload, target, settings.max_file_bytes, settings.max_request_bytes - total_bytes)
                inputs.append(target)

            config_path = settings.default_config
            if config is not None:
                if (config.content_type or "").lower() not in CONFIG_MEDIA_TYPES:
                    raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Configuration must be TOML text.")
                config_path = workspace_path / "configuration.toml"
                total_bytes += await _save_upload(
                    config, config_path, settings.max_file_bytes, settings.max_request_bytes - total_bytes
                )

            try:
                execution = run_pipeline_service(inputs, workspace_path / "artifacts", load_configuration(config_path))
            except (ConfigurationError, IngestionError) as error:
                LOGGER.info("Client input rejected: %s", type(error).__name__)
                raise ApiError(422, "INPUT_ERROR", "Configuration or CSV input is invalid.") from error
            except PipelineServiceError as error:
                LOGGER.error("Pipeline service failed: %s", type(error).__name__)
                raise ApiError(500, "PROCESSING_FAILED", "The request could not be processed.") from error

            zip_path = workspace_path / "umux-artifacts.zip"
            _write_zip(zip_path, execution.artifact_paths)
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename="umux-artifacts.zip",
                background=BackgroundTask(workspace.cleanup),
            )
        except Exception:
            workspace.cleanup()
            raise

    return app


async def _save_upload(upload: UploadFile, target: Path, max_file_bytes: int, remaining_request_bytes: int) -> int:
    if remaining_request_bytes < 1:
        raise ApiError(413, "REQUEST_TOO_LARGE", "Request exceeds the configured size limit.")
    written = 0
    try:
        with target.open("xb") as destination:
            while chunk := await upload.read(64 * 1024):
                written += len(chunk)
                if written > max_file_bytes:
                    raise ApiError(413, "FILE_TOO_LARGE", "An uploaded file exceeds the configured size limit.")
                if written > remaining_request_bytes:
                    raise ApiError(413, "REQUEST_TOO_LARGE", "Request exceeds the configured size limit.")
                destination.write(chunk)
    finally:
        await upload.close()
        if written > max_file_bytes or written > remaining_request_bytes:
            target.unlink(missing_ok=True)
    return written


def _write_zip(zip_path: Path, artifact_paths: object) -> None:
    paths = artifact_paths  # keep a narrow public adapter boundary around the pipeline execution.
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in EXPECTED_ARTIFACTS:
                path = paths[name]  # type: ignore[index]
                if not path.is_file():  # type: ignore[union-attr]
                    raise PipelineServiceError("Pipeline did not create a complete artifact bundle")
                archive.write(path, arcname=name)  # type: ignore[arg-type]
    except (OSError, zipfile.BadZipFile) as error:
        raise PipelineServiceError("Could not prepare artifact bundle") from error


app = create_app()
