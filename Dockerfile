FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.20 /uv /uvx /bin/

COPY requirements.txt .

RUN uv pip install --system --no-cache-dir -r requirements.txt 

WORKDIR /app

COPY /connectdeaf-faq-api/src .

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2"]
