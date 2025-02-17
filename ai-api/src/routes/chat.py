import json
import traceback

from config.logs import logger
from config.settings import settings
from fastapi import APIRouter, Depends, HTTPException
from infra.vector_database import AzureSearchVectorDB
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/faq", tags=["FAQ"])


class FaqRequest(BaseModel):
    user_message: str = Field(
        example="O que é o ConnectDeaf",
        description="Mensagem enviada pelo usuário para consulta",
    )


class FaqResponse(BaseModel):
    response: str = Field(
        example="ConnectDeaf é um marketplace que conecta surdos a profissionais capacitados para atendê-los.",
        description="Resposta gerada pelo FAQBot",
    )


class FaqUseCase:
    """
    Serviço de FAQ que utiliza Azure Search e OpenAI para gerar respostas baseadas em documentos fornecidos.
    """

    def __init__(self):
        """
        Inicializa o caso de uso de FAQ com Azure Search e OpenAI.
        """
        logger.info("Inicializando FaqUseCase.")
        self.az = AzureSearchVectorDB()
        self.openai_client = AsyncAzureOpenAI(
            api_key=settings.OPENAI_API_KEY,
            api_version=settings.OPENAI_API_VERSION,
            azure_endpoint=settings.OPENAI_AZURE_ENDPOINT,
        )

    async def _generate_response(self, query: str) -> FaqResponse:
        """
        Gera uma resposta para a consulta fornecida utilizando documentos similares e OpenAI.
        """
        try:
            logger.info(f"Processando consulta do usuário: '{query}'")

            logger.info("Buscando documentos similares utilizando Azure Search...")
            df = await self.az.search_similar_documents(
                query, k=3, filters="type eq 'faq'"
            )
            logger.info(
                f"Documentos similares retornados: {len(df['doc_content'].values)} documentos encontrados."
            )

            combined_content = " ".join(
                [
                    json.loads(doc).get("answer", "")
                    for doc in df["doc_content"].values
                    if isinstance(doc, str)
                ]
            )
            logger.info(
                f"Conteúdo combinado dos documentos: '{combined_content[:100]}...'"
            )

            system_prompt = (
                "Você é um assistente útil chamado FAQBot, responsável por responder perguntas de usuários com base nos documentos fornecidos. "
                "Você deve sempre se basear exclusivamente no conteúdo fornecido. Caso a resposta não esteja presente, responda educadamente que não sabe ou que a informação não está disponível. "
                "Seja direto, claro e profissional em suas respostas."
            )
            assistant_context = f"Aqui estão informações relevantes para responder à consulta: {combined_content}."
            logger.info(
                "Prompt e contexto do assistente definidos, enviando consulta para o OpenAI..."
            )

            response = await self.openai_client.chat.completions.create(
                temperature=0.3,
                model=settings.OPENAI_GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": assistant_context},
                    {"role": "user", "content": query},
                ],
            )

            final_response = response.choices[0].message.content.strip()
            logger.info(f"Resposta gerada pelo OpenAI: '{final_response}'")
            return FaqResponse(response=final_response)

        except Exception as e:
            logger.error(f"Erro ao processar a consulta: {e}")
            logger.error(traceback.format_exc())
            return FaqResponse(
                response="Desculpe, não consegui encontrar uma resposta para sua pergunta."
            )

    async def chat(self, query: str) -> FaqResponse:
        logger.info(f"Iniciando método chat com a consulta: '{query}'")
        return await self._generate_response(query)


@router.post("/chat", response_model=FaqResponse)
async def chat_faq(
    request: FaqRequest, faq_service: FaqUseCase = Depends(FaqUseCase)
) -> FaqResponse:
    """
    Endpoint para conversar com o serviço de FAQ.

    - **user_message**: A mensagem enviada pelo usuário.

    Retorna a resposta gerada pelo serviço de FAQ.
    """
    try:
        logger.info(f"Endpoint /chat recebido com mensagem: '{request.user_message}'")
        response = await faq_service.chat(request.user_message)
        logger.info("Resposta retornada com sucesso pelo endpoint /chat.")
        return response
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Erro ao processar a consulta no endpoint /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {e}")
