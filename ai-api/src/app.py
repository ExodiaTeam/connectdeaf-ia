from fastapi import FastAPI
from routes import certificate, chat

app = FastAPI()
app.include_router(certificate.router)
app.include_router(chat.router)
