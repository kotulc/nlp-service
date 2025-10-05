import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_tags():
    payload = {"content": "Sample text for tagging goes here Mr. Deloit."}
    response = client.post("/tags/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "result" in data
    assert "metadata" in data
    
    assert "tags" in data["result"]
    assert "scores" in data["result"]


def test_tags_required():
    """Verify response fails without required content"""
    response = client.post("/tags/", json={})
    assert response.status_code != 200


@pytest.mark.parametrize("min_length, max_length", [(1, 2), (2, 4), (3, 5)])
def test_tags_min_max(min_length, max_length):
    # 2. min/max length of returned related tags are respected
    payload = {
        "content": "Test content for tagging.",
        "min_length": min_length,
        "max_length": max_length,
        "tag": "related"
    }
    response = client.post("/tags/", json=payload)
    assert response.status_code == 200

    data = response.json()
    for tag in data["result"]["tags"]["related"]:
        word_count = len(tag.split())
        assert min_length <= word_count <= max_length


@pytest.mark.parametrize("top_n", [1, 3, 5, 10])
def test_tags_top_n_limit(top_n):
    # 3. All returned results have length <= top_n
    payload = {
        "content": "Test content for tagging.",
        "top_n": top_n,
        "tag": "related"
    }
    response = client.post("/tags/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    tags = data["result"].get("related", [])
    assert len(tags) <= top_n


from fastapi.testclient import TestClient
from app.main import app  


if __name__ == "__main__":
    test_tags()
    test_tags_required()
    test_tags_min_max(1, 2)
    test_tags_top_n_limit(5)
