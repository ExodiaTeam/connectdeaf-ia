from fastapi import FastAPI
from routes import faq

app = FastAPI()
app.include_router(faq.router)
