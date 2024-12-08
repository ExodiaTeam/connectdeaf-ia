import json
import uuid

import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from config.logs import logger
from config.settings import settings
from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse


class AzureSearchVectorDB:
    """
    Uma classe para interagir com um índice de pesquisa do Azure configurado para pesquisa vetorial.

    Esta classe:
      - Cria e configura um índice de pesquisa do Azure para pesquisas vetoriais.
      - Insere documentos junto com seus embeddings vetoriais.
      - Integra-se com o Azure OpenAI para gerar embeddings.
    """

    def __init__(self) -> None:
        """
        Inicializa a instância do banco de dados vetorial do Azure Search.
        """
        self.index_name = "faq-index"
        self.embeddings_dimension = 1536

        self.index_client = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
        )

        self.search_client = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=self.index_name,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
        )

        self.openai_client = AzureOpenAI(
            api_key=settings.OPENAI_API_KEY,
            api_version=settings.OPENAI_API_VERSION,
            azure_endpoint=settings.OPENAI_AZURE_ENDPOINT,
        )

        if self.index_name not in self._get_existing_index_names():
            self._create_vector_index()

    def insert_document_faq(self, question: str, answer: str) -> None:
        """
        Insere um documento no índice de pesquisa do Azure com embeddings gerados.

        Args:
            question (str): O texto da pergunta.
            answer (str): O texto da resposta.
        """
        try:
            logger.info(f"Inserting a document into index '{self.index_name}'...")

            data = {"question": question, "answer": answer}
            data_str = json.dumps(data)

            doc_embeddings = self._generate_embeddings(data_str)

            doc = {
                "id": str(uuid.uuid4()),
                "type": "faq",
                "doc_content": data_str,
                "doc_content_embeddings": doc_embeddings,
            }

            self.search_client.upload_documents(documents=[doc])
            logger.info("Document inserted successfully.")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")

    def insert_document(self, content: str) -> None:
        """
        Insere um documento no índice de pesquisa do Azure com embeddings gerados.

        Args:
            content (str): O conteúdo do documento.
        """
        try:
            logger.info(f"Inserting a document into index '{self.index_name}'...")

            doc_embeddings = self._generate_embeddings(content)

            doc = {
                "id": str(uuid.uuid4()),
                "type": "doc",
                "doc_content": content,
                "doc_content_embeddings": doc_embeddings,
            }

            self.search_client.upload_documents(documents=[doc])
            logger.info("Document inserted successfully.")
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")

    def search_similar_documents(self, query: str, k: int = 3, filters: str = None) -> pd.DataFrame:
        """
        Pesquisa documentos similares no índice de pesquisa do Azure.

        Args:
            query (str): O texto da consulta.
            k (int): O número de documentos similares a serem retornados.
            filters (str): Expressão de filtro OData opcional.

        Returns:
            pd.DataFrame: Um DataFrame contendo os documentos similares encontrados.
        """
        query_embedding = self._generate_embeddings(query)
        vector_query = VectorizedQuery(vector=query_embedding, fields="document_vector")
        results = self.search_client.search(
            top=k,
            vector_queries=[vector_query],
            select=["id", "doc_content", "type"],
            filter=filters if filters else "type eq 'doc'",
        )
        df = pd.DataFrame(results)
        return df

    def _create_vector_index(self) -> None:
        """
        Cria o índice vetorial do Azure Search se ele não existir.
        """
        logger.info(f"Creating index '{self.index_name}'...")

        fields = self._get_index_fields()
        vector_search = self._get_vector_search_config()

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        try:
            self.index_client.create_or_update_index(index)
            logger.info(f"Index '{self.index_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating index '{self.index_name}': {e}")

    def _get_index_fields(self) -> list:
        """
        Define o esquema do índice de pesquisa do Azure.

        Returns:
            list: Uma lista de campos do índice.
        """
        return [
            SearchableField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchField(
                name="type",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True,
            ),
            SearchableField(
                name="doc_content",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SearchField(
                name="document_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.embeddings_dimension,
                vector_search_profile_name="ExhaustiveKnnProfile",
            ),
        ]

    def _get_vector_search_config(self) -> VectorSearch:
        """
        Configura o algoritmo de pesquisa vetorial e perfis.

        Returns:
            VectorSearch: A configuração de pesquisa vetorial.
        """
        return VectorSearch(
            algorithm_configurations=[
                ExhaustiveKnnAlgorithmConfiguration(
                    name="ExhaustiveKnn",
                    kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                    parameters=ExhaustiveKnnParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE
                    ),
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="ExhaustiveKnnProfile",
                    algorithm_configuration_name="ExhaustiveKnn",
                )
            ],
        )

    def insert_documents_from_json(self, json_filepath: str) -> None:
        """
        Insere documentos de um arquivo JSON no índice de pesquisa do Azure.

        Args:
            json_filepath (str): O caminho para o arquivo JSON contendo os documentos.
        """
        try:
            with open(json_filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

            for item in data.get("faq", []):
                self.insert_document_faq(item["question"], item["answer"])

            logger.info("All documents from JSON file inserted successfully.")
        except Exception as e:
            logger.error(f"Failed to insert documents from JSON file: {e}")

    def _get_existing_index_names(self) -> list:
        """
        Recupera uma lista de nomes de índices existentes no Azure Search.

        Returns:
            list: Uma lista de nomes de índices existentes.
        """
        try:
            return [name for name in self.index_client.list_index_names()]
        except Exception as e:
            logger.error(f"Error retrieving existing index names: {e}")
            return []

    def _generate_embeddings(self, content: str) -> list[float] | list[list[float]]:
        """
        Gera embeddings para um determinado conteúdo usando o Azure OpenAI.

        Args:
            content (str): O conteúdo em string para gerar embeddings.

        Returns:
            list[float] | list[list[float]]: Uma lista de floats representando o vetor de embeddings.
        """
        logger.info("Generating embeddings for content.")
        try:
            response: CreateEmbeddingResponse = self.openai_client.embeddings.create(
                input=[content],
                model=settings.OPENAI_EMBEDDING_MODEL,
            )
            if len(response.data) == 1:
                return response.data[0].embedding
            else:
                return [record.embedding for record in response.data]

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
