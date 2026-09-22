from __future__ import annotations

import httpx
import pytest
from threading import Event

import server


class FakeAgent:
    device = "cpu"
    dtype = "torch.float32"
    cfg = {"max_len": 512, "head_max_len": 192}

    def predict(self, state, questions):
        assert state == {"body": "refund me"}
        assert questions["route"]["criteria"] == {
            "billing": "payments",
            "support": "everything else",
        }
        return {
            "model": "laya-rl-agent",
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "support": 0.1},
                    "confidence": 0.8,
                    "action": {"act_probability": 0.9},
                }
            },
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }


class FailingAgent(FakeAgent):
    def predict(self, state, questions):
        raise RuntimeError("synthetic inference failure")


class InvalidAgent(FakeAgent):
    def predict(self, state, questions):
        raise ValueError("invalid model input")


class InvalidUsageAgent(FakeAgent):
    def predict(self, state, questions):
        result = super().predict(state, questions)
        result["usage"]["input_tokens"] = True
        return result


class BlockingAgent(FakeAgent):
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def predict(self, state, questions):
        self.started.set()
        assert self.release.wait(timeout=2)
        return super().predict(state, questions)


def inference_request():
    return {
        "state": {"body": "refund me"},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Where should this go?",
                "criteria": {
                    "billing": "payments",
                    "support": "everything else",
                },
            }
        },
    }


@pytest.fixture(autouse=True)
def configured_api_key(monkeypatch):
    monkeypatch.setattr(server.settings, "api_key", "test-secret")


def client(*, authenticated: bool = True):
    headers = {"Authorization": "Bearer test-secret"} if authenticated else None
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url="http://testserver",
        headers=headers,
    )


def reset_runtime_counters():
    server.runtime.inflight = 0
    server.runtime.executing = 0
    server.runtime.completed_total = 0
    server.runtime.failed_total = 0
    server.runtime.queue_rejected_total = 0


@pytest.mark.anyio
async def test_health_and_readiness(monkeypatch):
    monkeypatch.setattr(server.runtime, "agent", None)
    async with client() as test_client:
        assert (await test_client.get("/health")).json() == {"status": "ok"}
        response = await test_client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"status": "loading"}


@pytest.mark.anyio
async def test_strict_readiness_rejects_device_drift(monkeypatch):
    monkeypatch.setattr(server.settings, "device", "cuda")
    monkeypatch.setattr(server.settings, "require_configured_device", True)
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    async with client() as test_client:
        response = await test_client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded"}


@pytest.mark.anyio
async def test_cuda_and_cuda_zero_are_compatible(monkeypatch):
    agent = FakeAgent()
    agent.device = "cuda:0"
    monkeypatch.setattr(server.settings, "device", "cuda")
    monkeypatch.setattr(server.settings, "require_configured_device", True)
    monkeypatch.setattr(server.runtime, "agent", agent)
    async with client() as test_client:
        response = await test_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.anyio
async def test_system_one(monkeypatch):
    reset_runtime_counters()
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(2))
    async with client() as test_client:
        response = await test_client.post("/v1/systemone", json=inference_request())
    server.runtime.executor.shutdown()
    assert response.status_code == 200
    assert response.json()["answers"]["route"]["choice"] == "billing"
    assert server.runtime.inflight == 0
    assert server.runtime.executing == 0
    assert server.runtime.completed_total == 1
    assert server.runtime.failed_total == 0


@pytest.mark.anyio
async def test_inference_failure_updates_counters_and_error_code(monkeypatch):
    reset_runtime_counters()
    monkeypatch.setattr(server.runtime, "agent", FailingAgent())
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(2))
    async with client() as test_client:
        response = await test_client.post("/v1/systemone", json=inference_request())
    server.runtime.executor.shutdown()
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "inference_failed"
    assert server.runtime.inflight == 0
    assert server.runtime.executing == 0
    assert server.runtime.completed_total == 0
    assert server.runtime.failed_total == 1


@pytest.mark.anyio
async def test_model_value_error_updates_failure_counter(monkeypatch):
    reset_runtime_counters()
    monkeypatch.setattr(server.runtime, "agent", InvalidAgent())
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(2))
    async with client() as test_client:
        response = await test_client.post("/v1/systemone", json=inference_request())
    server.runtime.executor.shutdown()
    assert response.status_code == 422
    assert server.runtime.failed_total == 1


@pytest.mark.anyio
async def test_malformed_usage_fails_closed_and_updates_failure_counter(monkeypatch):
    reset_runtime_counters()
    monkeypatch.setattr(server.runtime, "agent", InvalidUsageAgent())
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(2))
    async with client() as test_client:
        response = await test_client.post("/v1/systemone", json=inference_request())
    server.runtime.executor.shutdown()
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "inference_failed"
    assert server.runtime.completed_total == 0
    assert server.runtime.failed_total == 1


@pytest.mark.anyio
async def test_queue_timeout_updates_rejection_counter(monkeypatch):
    reset_runtime_counters()
    monkeypatch.setattr(server.settings, "queue_timeout", 0.001)
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(0))
    async with client() as test_client:
        response = await test_client.post("/v1/systemone", json=inference_request())
    server.runtime.executor.shutdown()
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "queue_timeout"
    assert server.runtime.queue_rejected_total == 1
    assert server.runtime.inflight == 0


@pytest.mark.anyio
async def test_queue_capacity_is_hard_bounded(monkeypatch):
    reset_runtime_counters()
    monkeypatch.setattr(server.settings, "concurrency", 1)
    monkeypatch.setattr(server.settings, "queue_capacity", 1)
    monkeypatch.setattr(server.runtime, "inflight", 2)
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(2))
    async with client() as test_client:
        response = await test_client.post("/v1/systemone", json=inference_request())
    server.runtime.executor.shutdown()
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "queue_full"
    assert server.runtime.queue_rejected_total == 1
    assert server.runtime.inflight == 2


@pytest.mark.anyio
async def test_cancellation_keeps_capacity_until_executor_finishes(monkeypatch):
    reset_runtime_counters()
    agent = BlockingAgent()
    monkeypatch.setattr(server.runtime, "agent", agent)
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(2))
    request = server.SystemOneRequest.model_validate(inference_request())
    task = server.asyncio.create_task(server._predict(request))
    assert await server.asyncio.to_thread(agent.started.wait, 1)
    task.cancel()
    await server.asyncio.sleep(0.01)
    assert server.runtime.inflight == 1
    assert server.runtime.executing == 1
    agent.release.set()
    with pytest.raises(server.asyncio.CancelledError):
        await task
    server.runtime.executor.shutdown()
    assert server.runtime.inflight == 0
    assert server.runtime.executing == 0
    assert server.runtime.completed_total == 1


@pytest.mark.anyio
async def test_rejects_invalid_choice():
    payload = inference_request()
    payload["questions"]["route"]["criteria"] = {"only": "one option"}
    async with client() as test_client:
        response = await test_client.post("/v1/systemone", json=payload)
    assert response.status_code == 422


@pytest.mark.anyio
async def test_api_key(monkeypatch):
    monkeypatch.setattr(server.settings, "api_key", "test-secret")
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    async with client(authenticated=False) as test_client:
        assert (await test_client.get("/v1/model-info")).status_code == 401

        bearer_response = await test_client.get(
            "/v1/model-info",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert bearer_response.status_code == 200
        assert bearer_response.json()["max_context"] == 512

        header_response = await test_client.get(
            "/v1/model-info",
            headers={"X-API-Key": "test-secret"},
        )
        assert header_response.status_code == 200


@pytest.mark.anyio
async def test_worker_status_is_authenticated_and_reports_runtime(monkeypatch):
    reset_runtime_counters()
    monkeypatch.setattr(server.settings, "api_key", "test-secret")
    monkeypatch.setattr(server.settings, "worker_id", "laya-test")
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    async with client(authenticated=False) as test_client:
        assert (await test_client.get("/v1/worker-status")).status_code == 401
        response = await test_client.get(
            "/v1/worker-status",
            headers={"Authorization": "Bearer test-secret"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["protocol_version"] == 1
    assert body["status"] == "ready"
    assert body["worker_id"] == "laya-test"
    assert body["actual_device"] == "cpu"
    assert body["capacity"] == server.settings.concurrency
    assert body["inflight"] == 0
    assert body["executing"] == 0
    assert body["queued"] == 0
    assert body["completed_total"] == 0
    assert body["failed_total"] == 0
    assert body["queue_rejected_total"] == 0
    assert body["uptime_seconds"] >= 0
    assert len(body["instance_id"]) == 36


@pytest.mark.anyio
async def test_unconfigured_api_key_fails_closed(monkeypatch):
    monkeypatch.setattr(server.settings, "api_key", None)
    async with client(authenticated=False) as test_client:
        response = await test_client.get("/v1/worker-status")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "api_key_not_configured"


@pytest.mark.anyio
async def test_correlation_headers_cover_success_and_validation_errors(monkeypatch):
    monkeypatch.setattr(server.settings, "worker_id", "laya-test")
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    async with client() as test_client:
        supplied = await test_client.get("/health", headers={"X-Request-ID": "req-123"})
        invalid = await test_client.post("/v1/systemone", json={"bad": "payload"})
    for response in (supplied, invalid):
        assert response.headers["X-Laya-Worker-ID"] == "laya-test"
        assert response.headers["X-Laya-Worker-Instance"] == server.runtime.instance_id
        assert response.headers["X-Laya-Protocol-Version"] == "1"
    assert supplied.headers["X-Request-ID"] == "req-123"
    assert len(invalid.headers["X-Request-ID"]) == 36


@pytest.mark.anyio
async def test_unhandled_error_has_structured_protocol_response(monkeypatch):
    monkeypatch.setattr(server.settings, "worker_id", "laya-test")
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    original_version = server.metadata.version

    def broken_version(package):
        if package == "laya":
            raise RuntimeError("synthetic metadata failure")
        return original_version(package)

    monkeypatch.setattr(server.metadata, "version", broken_version)
    async with client() as test_client:
        response = await test_client.get(
            "/v1/worker-status",
            headers={"X-Request-ID": "req-error"},
        )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "internal_error"
    assert response.headers["X-Request-ID"] == "req-error"
    assert response.headers["X-Laya-Protocol-Version"] == "1"


@pytest.mark.parametrize("worker_id", ["", "bad\nheader", "x" * 129])
def test_runtime_settings_reject_invalid_worker_id(monkeypatch, worker_id):
    monkeypatch.setattr(server.settings, "worker_id", worker_id)
    with pytest.raises(RuntimeError, match="LAYA_WORKER_ID"):
        server._validate_runtime_settings()


def test_openapi_exposes_auth_controls():
    schema = server.app.openapi()
    schemes = schema["components"]["securitySchemes"]
    assert schemes["BearerAuth"] == {
        "type": "http",
        "description": "Enter the Laya API token without the `Bearer` prefix.",
        "scheme": "bearer",
    }
    assert schemes["ApiKeyAuth"] == {
        "type": "apiKey",
        "description": "Enter the Laya API token.",
        "in": "header",
        "name": "X-API-Key",
    }
    security = schema["paths"]["/v1/systemone"]["post"]["security"]
    assert {"BearerAuth": []} in security
    assert {"ApiKeyAuth": []} in security
