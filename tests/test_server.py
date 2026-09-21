from __future__ import annotations

from fastapi.testclient import TestClient

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


def test_health_and_readiness(monkeypatch):
    monkeypatch.setattr(server.runtime, "agent", None)
    client = TestClient(server.app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_system_one(monkeypatch):
    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    monkeypatch.setattr(server.runtime, "executor", server.ThreadPoolExecutor(1))
    monkeypatch.setattr(server.runtime, "execution_slots", server.asyncio.Semaphore(1))
    monkeypatch.setattr(server.runtime, "admission_slots", server.asyncio.Semaphore(2))
    client = TestClient(server.app)
    response = client.post("/v1/systemone", json=inference_request())
    server.runtime.executor.shutdown()
    assert response.status_code == 200
    assert response.json()["answers"]["route"]["choice"] == "billing"


def test_rejects_invalid_choice():
    client = TestClient(server.app)
    payload = inference_request()
    payload["questions"]["route"]["criteria"] = {"only": "one option"}
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422


def test_api_key(monkeypatch):
    monkeypatch.setattr(server.settings, "api_key", "test-secret")
    client = TestClient(server.app)
    assert client.get("/v1/model-info").status_code == 401

    monkeypatch.setattr(server.runtime, "agent", FakeAgent())
    response = client.get(
        "/v1/model-info",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200
    assert response.json()["max_context"] == 512
