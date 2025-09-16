from fastapi.testclient import TestClient
from app.main import app  


client = TestClient(app)

def test_get_tags():
    payload = {"content": "Sample text to tags goes Mr. Deloit here."}
    response = client.post("/tags/", json=payload)
    assert response.status_code == 200

    data = response.json()
    print(data)


if __name__ == "__main__":
    test_get_tags()
