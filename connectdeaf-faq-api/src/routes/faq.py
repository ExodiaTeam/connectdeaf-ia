from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.faq import FAQService

router = APIRouter(prefix="/api/faq", tags=["FAQ"])


class FaqRequest(BaseModel):
    user_message: str


class FaqResponse(BaseModel):
    response: str


@router.post("/chat", response_model=FaqResponse)
async def chat_faq(request: FaqRequest, faq_service: FAQService = Depends(FAQService)) -> dict:
    """
    Conversar com o serviço de FAQ.

    Args:
        request (FaqRequest): A mensagem do usuário.
        faq_service (FAQService): O serviço de FAQ.

    Returns:
        dict: A resposta gerada pelo serviço de FAQ.
    """
    try:
        return faq_service.chat(request.user_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {e}")
