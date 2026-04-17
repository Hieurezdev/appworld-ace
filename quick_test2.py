from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordRequestForm

app = FastAPI()
@app.post("/auth/token")
def login(data: OAuth2PasswordRequestForm = Depends()):
    return {"ok": 1}

client = TestClient(app)
client.post("/auth/token", data={"username": "a", "password": None})
