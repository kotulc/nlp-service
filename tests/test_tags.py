from fastapi.testclient import TestClient
from app.main import app  


client = TestClient(app)

def test_tags():
    payload = {"content": "Sample text for tagging goes here Mr. Deloit."}
    response = client.post("/tags/", json=payload)
    assert response.status_code == 200

    data = response.json()
    print(data)


if __name__ == "__main__":
    test_tags()
