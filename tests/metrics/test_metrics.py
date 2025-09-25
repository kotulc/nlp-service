import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.metrics.metrics import MetricsType


client = TestClient(app)


def test_metrics_content_required():
    """Confirm only 'content' argument is required"""
    payload = {"content": "Test content for metrics."}
    response = client.post("/metrics/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "result" in data
    assert "metadata" in data

    for metric in MetricsType:
        assert metric.name.lower() in data["result"]


@pytest.mark.parametrize("metric_type", [m.name for m in MetricsType])
def test_metrics_type_parameterized(metric_type):
    """Test each metric type individually"""
    payload = {
        "content": "Test content for metrics.",
        "metrics": [metric_type]
    }
    response = client.post("/metrics/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    # Should contain the requested metric type in the response data
    assert metric_type.lower() in data["data"]


@pytest.mark.parametrize("metric_types", [
    [m.name for m in MetricsType],
    [MetricsType.__members__[k].name for k in list(MetricsType.__members__)[:2]],
])
def test_metrics_multiple_types(metric_types):
    """Test multiple metric types at once"""
    payload = {
        "content": "Test content for metrics.",
        "metrics": metric_types
    }
    response = client.post("/metrics/", json=payload)
    assert response.status_code == 200
    data = response.json()
    for metric in metric_types:
        assert metric.lower() in data["data"]


if __name__ == "__main__":
    test_metrics_content_required()
