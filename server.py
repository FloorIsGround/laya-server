"""Internal HTTP inference service for a single resident Laya checkpoint."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Annotated, Any, Literal

import laya
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator

LOGGER = logging.getLogger("laya.api")
MAX_QUESTIONS = 32
MAX_OPTIONS = 20
MAX_STATE_BYTES = 256 * 1024


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be at least one")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


class Settings:
    model = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
    subfolder = os.getenv("LAYA_SUBFOLDER") or None
    device = os.getenv("LAYA_DEVICE", "cpu")
    api_key = os.getenv("LAYA_API_KEY") or None
    concurrency = _positive_int("LAYA_CONCURRENCY", 1)
    queue_capacity = _positive_int("LAYA_QUEUE_CAPACITY", 64)
    queue_timeout = _positive_float("LAYA_QUEUE_TIMEOUT_SECONDS", 30.0)
    torch_threads = _positive_int("LAYA_TORCH_THREADS", 12)
    worker_id = os.getenv("LAYA_WORKER_ID", "laya-worker")
    require_configured_device = _boolean("LAYA_REQUIRE_CONFIGURED_DEVICE", True)


settings = Settings()


def _validate_runtime_settings() -> None:
    worker_id = settings.worker_id
    if not (1 <= len(worker_id) <= 128) or not all(32 <= ord(char) < 127 for char in worker_id):
        raise RuntimeError("LAYA_WORKER_ID must be 1-128 printable ASCII characters")
    if not settings.api_key:
        raise RuntimeError("LAYA_API_KEY is required")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Usage(StrictModel):
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)


class InferenceProtocolError(RuntimeError):
    pass


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: str = Field(min_length=1, max_length=4_000)
    criteria: dict[str, Any] | list[str]

    @model_validator(mode="after")
    def validate_criteria(self) -> "ChoiceQuestion":
        count = len(self.criteria)
        if not 2 <= count <= MAX_OPTIONS:
            raise ValueError(f"choice criteria must contain 2 to {MAX_OPTIONS} options")
        if isinstance(self.criteria, dict) and any(not key.strip() for key in self.criteria):
            raise ValueError("choice option keys must not be empty")
        return self


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: str = Field(min_length=1, max_length=4_000)
    criteria: list[Any]

    @model_validator(mode="after")
    def validate_criteria(self) -> "ScoreQuestion":
        if not 2 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"score criteria must contain 2 to {MAX_OPTIONS} levels")
        return self


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: str = Field(min_length=1, max_length=4_000)
    criteria: dict[str, Any] | None = None


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]
State = str | dict[str, Any] | list[Any]


class SystemOneRequest(StrictModel):
    state: State
    questions: dict[str, Question]

    @model_validator(mode="after")
    def validate_request(self) -> "SystemOneRequest":
        if not 1 <= len(self.questions) <= MAX_QUESTIONS:
            raise ValueError(f"questions must contain 1 to {MAX_QUESTIONS} entries")
        if any(not key.strip() or len(key) > 128 for key in self.questions):
            raise ValueError("question IDs must be non-empty and at most 128 characters")
        state_bytes = json.dumps(
            self.state,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(state_bytes) > MAX_STATE_BYTES:
            raise ValueError(f"serialized state exceeds {MAX_STATE_BYTES} bytes")
        return self

    def laya_questions(self) -> dict[str, dict[str, Any]]:
        return {
            key: question.model_dump(exclude_none=True)
            for key, question in self.questions.items()
        }


class Runtime:
    agent: Any | None = None
    executor: ThreadPoolExecutor | None = None
    execution_slots: asyncio.Semaphore | None = None
    admission_slots: asyncio.Semaphore | None = None
    instance_id: str = str(uuid.uuid4())
    started_at: float = time.monotonic()
    inflight: int = 0
    executing: int = 0
    completed_total: int = 0
    failed_total: int = 0
    queue_rejected_total: int = 0


runtime = Runtime()


def _service_version() -> str:
    try:
        return metadata.version("laya-api")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _device_matches(desired: str, actual: str) -> bool:
    desired_normalized = desired.strip().lower()
    actual_normalized = actual.strip().lower()
    if desired_normalized == actual_normalized:
        return True
    if desired_normalized == "cuda" and actual_normalized.startswith("cuda:"):
        return True
    return False


def _device_degraded() -> bool:
    return bool(
        settings.require_configured_device
        and runtime.agent is not None
        and not _device_matches(settings.device, str(runtime.agent.device))
    )


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _validate_inference_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise InferenceProtocolError("Laya inference result must be an object")
    try:
        Usage.model_validate(result.get("usage"))
    except ValidationError as exc:
        raise InferenceProtocolError("Laya inference result contains invalid usage") from exc
    return result


def _load_agent() -> Any:
    torch.set_num_threads(settings.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work starts.
        pass
    return laya.load(
        settings.model,
        device=settings.device,
        subfolder=settings.subfolder,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _validate_runtime_settings()
    runtime.instance_id = str(uuid.uuid4())
    runtime.started_at = time.monotonic()
    runtime.inflight = 0
    runtime.executing = 0
    runtime.completed_total = 0
    runtime.failed_total = 0
    runtime.queue_rejected_total = 0
    LOGGER.info(
        "loading Laya model=%s subfolder=%s device=%s",
        settings.model,
        settings.subfolder,
        settings.device,
    )
    runtime.execution_slots = asyncio.Semaphore(settings.concurrency)
    runtime.admission_slots = asyncio.Semaphore(
        settings.concurrency + settings.queue_capacity
    )
    runtime.executor = ThreadPoolExecutor(
        max_workers=settings.concurrency,
        thread_name_prefix="laya-inference",
    )
    runtime.agent = await asyncio.to_thread(_load_agent)
    LOGGER.info("Laya model ready on %s", runtime.agent.device)
    try:
        yield
    finally:
        runtime.agent = None
        if runtime.executor is not None:
            runtime.executor.shutdown(wait=True, cancel_futures=True)
        runtime.executor = None


app = FastAPI(
    title="Laya Internal Inference API",
    version="1.0.0",
    description=(
        "Open **Authorize** and enter the token under either `BearerAuth` "
        "(token only, without the `Bearer` prefix) or `ApiKeyAuth`. "
        "The same token works in the `Authorization: Bearer *** and "
        "`X-API-Key: *** headers."
    ),
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)


@app.middleware("http")
async def correlation_headers(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    if 1 <= len(supplied) <= 128 and all(32 <= ord(char) < 127 for char in supplied):
        request_id = supplied
    else:
        request_id = str(uuid.uuid4())
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 - convert unhandled application faults to protocol errors
        LOGGER.exception("unhandled Laya API error")
        response = JSONResponse(
            {"detail": _error("internal_error", "internal server error")},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Laya-Worker-ID"] = settings.worker_id
    response.headers["X-Laya-Worker-Instance"] = runtime.instance_id
    response.headers["X-Laya-Protocol-Version"] = "1"
    return response


bearer = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description="Enter the Laya API token without the `Bearer` prefix.",
)
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    scheme_name="ApiKeyAuth",
    description="Enter the Laya API token.",
)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer),
    api_key: str | None = Security(api_key_header),
) -> None:
    if settings.api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error("api_key_not_configured", "worker API key is not configured"),
        )
    bearer_token = ""
    if credentials is not None and credentials.scheme.lower() == "bearer":
        bearer_token = credentials.credentials
    header_token = api_key or ""
    if not (
        hmac.compare_digest(bearer_token, settings.api_key)
        or hmac.compare_digest(header_token, settings.api_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _require_ready() -> Any:
    if runtime.agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error("model_not_ready", "model is not ready"),
        )
    if _device_degraded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error("device_degraded", "runtime device differs from configured device"),
        )
    return runtime.agent


async def _predict(request: SystemOneRequest) -> dict[str, Any]:
    agent = _require_ready()
    if (
        runtime.admission_slots is None
        or runtime.execution_slots is None
        or runtime.executor is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error("model_not_ready", "inference runtime is not ready"),
        )

    maximum_inflight = settings.concurrency + settings.queue_capacity
    if runtime.inflight >= maximum_inflight:
        runtime.queue_rejected_total += 1
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error("queue_full", "inference queue is full"),
            headers={"Retry-After": "1"},
        )

    runtime.inflight += 1
    admitted = False
    try:
        try:
            await asyncio.wait_for(
                runtime.admission_slots.acquire(),
                timeout=settings.queue_timeout,
            )
            admitted = True
        except TimeoutError as exc:
            runtime.queue_rejected_total += 1
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_error("queue_timeout", "inference queue is full"),
                headers={"Retry-After": "1"},
            ) from exc

        async with runtime.execution_slots:
            runtime.executing += 1
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                runtime.executor,
                agent.predict,
                request.state,
                request.laya_questions(),
            )
            try:
                result = _validate_inference_result(await asyncio.shield(future))
            except asyncio.CancelledError:
                try:
                    _validate_inference_result(await asyncio.shield(future))
                except Exception:
                    runtime.failed_total += 1
                else:
                    runtime.completed_total += 1
                raise
            except ValueError as exc:
                runtime.failed_total += 1
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=str(exc),
                ) from exc
            except Exception as exc:
                runtime.failed_total += 1
                LOGGER.exception("Laya inference failed")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=_error("inference_failed", "inference failed"),
                ) from exc
            else:
                runtime.completed_total += 1
                return result
            finally:
                runtime.executing -= 1
    finally:
        if admitted:
            runtime.admission_slots.release()
        runtime.inflight -= 1


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["operations"])
async def ready(response: Response) -> dict[str, str]:
    if runtime.agent is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading"}
    if _device_degraded():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded"}
    return {"status": "ready"}


@app.get(
    "/v1/worker-status",
    dependencies=[Depends(require_api_key)],
    tags=["operations"],
)
async def worker_status() -> dict[str, Any]:
    if runtime.agent is None:
        worker_state = "loading"
        actual_device = None
        dtype = None
    else:
        worker_state = "degraded" if _device_degraded() else "ready"
        actual_device = str(runtime.agent.device)
        dtype = str(runtime.agent.dtype)
    return {
        "protocol_version": 1,
        "status": worker_state,
        "worker_id": settings.worker_id,
        "instance_id": runtime.instance_id,
        "service_version": _service_version(),
        "laya_version": metadata.version("laya"),
        "model": settings.model,
        "desired_device": settings.device,
        "actual_device": actual_device,
        "dtype": dtype,
        "capacity": settings.concurrency,
        "queue_capacity": settings.queue_capacity,
        "inflight": runtime.inflight,
        "executing": runtime.executing,
        "queued": max(runtime.inflight - runtime.executing, 0),
        "completed_total": runtime.completed_total,
        "failed_total": runtime.failed_total,
        "queue_rejected_total": runtime.queue_rejected_total,
        "uptime_seconds": max(int(time.monotonic() - runtime.started_at), 0),
    }


@app.get(
    "/v1/model-info",
    dependencies=[Depends(require_api_key)],
    tags=["inference"],
)
async def model_info() -> dict[str, Any]:
    agent = _require_ready()
    return {
        "model": settings.model,
        "subfolder": settings.subfolder,
        "laya_version": metadata.version("laya"),
        "device": str(agent.device),
        "dtype": str(agent.dtype),
        "max_context": agent.cfg.get("max_len"),
        "head_max_length": agent.cfg.get("head_max_len"),
        "workers_per_process": settings.concurrency,
        "torch_threads": torch.get_num_threads(),
        "api_version": "1.0.0",
    }


@app.post(
    "/v1/systemone",
    dependencies=[Depends(require_api_key)],
    tags=["inference"],
)
async def system_one(request: SystemOneRequest) -> dict[str, Any]:
    return await _predict(request)


@app.post(
    "/v1/evaluate",
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
async def evaluate_alias(request: SystemOneRequest) -> dict[str, Any]:
    return await _predict(request)
