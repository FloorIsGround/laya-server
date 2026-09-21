"""Internal HTTP inference service for a single resident Laya checkpoint."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Annotated, Any, Literal

import laya
import torch
from fastapi import Depends, FastAPI, HTTPException, Response, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class Settings:
    model = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
    subfolder = os.getenv("LAYA_SUBFOLDER") or None
    device = os.getenv("LAYA_DEVICE", "cpu")
    api_key = os.getenv("LAYA_API_KEY") or None
    concurrency = _positive_int("LAYA_CONCURRENCY", 1)
    queue_capacity = _positive_int("LAYA_QUEUE_CAPACITY", 64)
    queue_timeout = _positive_float("LAYA_QUEUE_TIMEOUT_SECONDS", 30.0)
    torch_threads = _positive_int("LAYA_TORCH_THREADS", 12)


settings = Settings()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


runtime = Runtime()


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
        "The same token works in the `Authorization: Bearer ...` and "
        "`X-API-Key: ...` headers."
    ),
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)
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
        return
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
            detail="model is not ready",
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
            detail="inference runtime is not ready",
        )

    try:
        await asyncio.wait_for(
            runtime.admission_slots.acquire(),
            timeout=settings.queue_timeout,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference queue is full",
            headers={"Retry-After": "1"},
        ) from exc

    try:
        async with runtime.execution_slots:
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(
                    runtime.executor,
                    agent.predict,
                    request.state,
                    request.laya_questions(),
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
    finally:
        runtime.admission_slots.release()


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["operations"])
async def ready(response: Response) -> dict[str, str]:
    if runtime.agent is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "loading"}
    return {"status": "ready"}


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
