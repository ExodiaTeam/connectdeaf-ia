import json

from config.logs import logger
from config.settings import settings
from infra.vector_database import AzureSearchVectorDB
from openai import AzureOpenAI


class FAQService:
    """
    Serviço de FAQ que utiliza Azure Search e OpenAI para gerar respostas baseadas em documentos fornecidos.
    """

    def __init__(self):
        """
        Inicializa o serviço de FAQ com Azure Search e OpenAI.
        """
        self.az = AzureSearchVectorDB()
        self.openai_client = AzureOpenAI(
            api_key=settings.OPENAI_API_KEY,
            api_version=settings.OPENAI_API_VERSION,
            azure_endpoint=settings.OPENAI_AZURE_ENDPOINT,
        )

    def _generate_response(self, query: str) -> dict:
        """
        Gera uma resposta para a consulta fornecida utilizando documentos similares e OpenAI.

        Args:
            query (str): A consulta do usuário.

        Returns:
            dict: Um dicionário contendo a resposta gerada.
        """
        try:
            logger.info(f"Chatting with the user: {query}")
            df = self.az.search_similar_documents(query, k=3, filters="type eq 'faq'")
            combined_content = " ".join(
                [
                    json.loads(doc)["answer"]
                    for doc in df["doc_content"].values
                    if isinstance(doc, str)
                ]
            )

            response = self.openai_client.chat.completions.create(
                temperature=0.0,
                model=settings.OPENAI_GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente útil que responde exclusivamente com base nos documentos fornecidos. "
                            "Se não souber a resposta, diga 'Não sei' ou 'Essa informação não está disponível'."
                        ),
                    },
                    {"role": "assistant", "content": f"Contexto: {combined_content}"},
                    {"role": "user", "content": query},
                ],
            )

            return {"response": response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Failed to chat with the user: {e}")
            return {"response": "Desculpe, não consegui encontrar uma resposta para sua pergunta."}

    def chat(self, query: str) -> dict:
        """
        Método público para gerar uma resposta para a consulta fornecida.

        Args:
            query (str): A consulta do usuário.

        Returns:
            dict: Um dicionário contendo a resposta gerada.
        """
        return self._generate_response(query)
