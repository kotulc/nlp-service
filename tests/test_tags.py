from fastapi.testclient import TestClient
from app.main import app  


client = TestClient(app)

def test_get_tags():
    payload = {
        "content": "Sample text to tag goes here.",
        "tag": "all_tags",
        "merge": False,
        "commit": False
    }
    response = client.post("/tags/", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "data" in data  # Adjust based on your BaseResponse schema


if __name__ == "__main__":
    test_get_tags()
