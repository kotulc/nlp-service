import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.summary.summary import SUMMARY_TYPES


client = TestClient(app)


def test_summary():
    # 1. Only 'content' is required
    payload = {"content": "Test content for summary."}
    response = client.post("/summary/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    assert data["data"]  # Should return some summary


@pytest.mark.parametrize("summary_type", list(SUMMARY_TYPES.keys()))
def test_summary_type(summary_type):
    # 2. Test each summary type individually
    payload = {
        "content": "Test content for summary.",
        "summary": summary_type
    }
    response = client.post("/summary/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "data" in data
    # Should contain the requested summary type in the response data
    assert summary_type in data["data"]


@pytest.mark.parametrize("top_n", [1, 3, 5])
def test_summary_top_n(top_n):
    # 3. All returned results have length <= top_n
    payload = {
        "content": "Test content for summary.",
        "summary": "description",
        "top_n": top_n
    }
    response = client.post("/summary/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    summaries = data["data"].get("description", [])
    assert len(summaries) <= top_n


if __name__ == "__main__":
    test_summary()
    test_summary_type("outline")
    test_summary_top_n(top_n=5)
