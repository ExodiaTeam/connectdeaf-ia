import base64
import traceback

from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from config.logs import logger
from config.settings import settings
from fastapi import APIRouter, Depends, HTTPException
from infra.storage import Storage
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/api/documents", tags=["Documentos"])


class UploadDocumentRequest(BaseModel):
    filename: str = Field(
        description="Nome do arquivo", examples=["file.pdf"]
    )
    content: str = Field(
        ..., description="Conteúdo do arquivo em base64", examples=["base64_string"]
    )

    @field_validator("content")
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("O conteúdo não é uma string base64 válida.") from e
        return v

    @field_validator("filename")
    def validate_filename(cls, v: str) -> str:
        if not v:
            raise ValueError("O nome do arquivo não pode ser vazio.")
        return v


class UploadDocumentResponse(BaseModel):
    response: str = Field(
        description="Referência ao arquivo no storage, ex: 'documents/file.pdf'",
        examples=["documents/file.pdf"],
    )


class VerifyDocumentRequest(BaseModel):
    document_path: str = Field(
        description="Caminho do arquivo no storage", examples=["documents/file.pdf"]
    )
    professional_name: str = Field(
        description="Nome do profissional a ser comparado com o certificado",
        examples=["João da Silva"],
    )

    @field_validator("document_path")
    def validate_document_path(cls, v: str) -> str:
        if not v:
            raise ValueError("O caminho do arquivo não pode ser vazio.")
        return v

    @field_validator("professional_name")
    def validate_professional_name(cls, v: str) -> str:
        if not v:
            raise ValueError("O nome do profissional não pode ser vazio.")
        return v


class VerifyDocumentResponse(BaseModel):
    response: str = Field(
        description="Resposta da verificação do arquivo",
        examples=["Válido", "Inválido"],
    )


class UploadFileUseCase:
    def __init__(self):
        self.storage = Storage()

    async def execute(self, request: UploadDocumentRequest) -> UploadDocumentResponse:
        logger.info("Executando caso de uso: UploadFileUseCase.")
        content_bytes = base64.b64decode(request.content)
        try:
            uploaded_path = await self.storage.upload_file(
                settings.DOCUMENTS_CONTAINER_NAME, request.filename, content_bytes
            )
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Erro no upload: {e}")
        logger.info(f"Arquivo carregado e disponível em: {uploaded_path}")
        return UploadDocumentResponse(response=uploaded_path)


class VerifyFileUseCase:
    def __init__(self):
        self.storage = Storage()
        self.openai_client = AsyncAzureOpenAI(
            api_key=settings.OPENAI_API_KEY,
            api_version=settings.OPENAI_API_VERSION,
            azure_endpoint=settings.OPENAI_AZURE_ENDPOINT,
        )
        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=settings.AZURE_OCR_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_OCR_KEY),
        )

    async def _ocr(self, doc_content: bytes) -> str:
        logger.info("Iniciando extração de texto via OCR.")
        try:
            async with self.document_analysis_client:
                poller = await self.document_analysis_client.begin_analyze_document(
                    "prebuilt-read", doc_content
                )
                result = await poller.result()
                extracted_text = result.content
                logger.info("Extração de texto via OCR concluída com sucesso.")
                return extracted_text
        except AzureError as e:
            logger.error(traceback.format_exc())
            logger.error(f"Erro ao realizar OCR: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao realizar OCR: {e}")

    async def _generate_response(
        self, extracted_text: str, professional_name: str
    ) -> str:
        logger.info("Gerando resposta via OpenAI com base no texto extraído.")
        system_prompt = (
            "Você é um assistente especializado em validação de certificados. "
            "Analise o conteúdo extraído do documento e verifique se ele contém todas as informações necessárias para que o certificado seja considerado válido. "
            "Considere o nome do profissional, o número do registro e a data de validade. "
            f"O nome do profissional informado é '{professional_name}'. "
            "Se todas as informações estiverem corretas e consistentes, responda 'Válido'. "
            "Caso contrário, responda 'Inválido'."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extracted_text},
        ]
        try:
            response = await self.openai_client.chat.completions.create(
                temperature=0.3,
                model=settings.OPENAI_GPT_MODEL,
                messages=messages,
            )
            result_text = response.choices[0].message.content.strip()
            logger.info("Resposta gerada com sucesso via OpenAI.")
            return result_text
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Erro ao gerar resposta via OpenAI: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao gerar resposta via OpenAI: {e}")

    async def execute(self, request: VerifyDocumentRequest) -> VerifyDocumentResponse:
        logger.info("Executando caso de uso: VerifyFileUseCase.")
        content_bytes = await self.storage.download_file(request.document_path)
        if not content_bytes:
            logger.error("Conteúdo do arquivo não foi encontrado.")
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")

        extracted_text = await self._ocr(content_bytes)
        if not extracted_text.strip():
            logger.error("Nenhum texto foi extraído do documento.")
            raise HTTPException(status_code=500, detail="Falha na extração de texto")

        verification_result = await self._generate_response(
            extracted_text, request.professional_name
        )
        logger.info(f"Verificação concluída com o resultado: {verification_result}")
        return VerifyDocumentResponse(response=verification_result)


@router.post("/upload_file", response_model=UploadDocumentResponse, status_code=201)
async def upload_file_route(
    request: UploadDocumentRequest,
    upload_usecase: UploadFileUseCase = Depends(UploadFileUseCase),
) -> UploadDocumentResponse:
    """
    Endpoint para fazer upload de um arquivo para o Azure Storage.

    - **filename**: Nome do arquivo a ser salvo.
    - **content**: Conteúdo do arquivo em base64.

    Retorna a referência do arquivo salvo no storage.
    """
    try:
        return await upload_usecase.execute(request)
    except Exception as e:
        logger.error(f"Erro no endpoint upload_file: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed: {e}")


@router.post("/verify_file", response_model=VerifyDocumentResponse)
async def verify_file_route(
    request: VerifyDocumentRequest,
    verify_usecase: VerifyFileUseCase = Depends(VerifyFileUseCase),
) -> VerifyDocumentResponse:
    """
    Endpoint para verificar se um certificado é válido.

    - **document_path**: Caminho do arquivo no storage, ex: 'documents/file.pdf'.
    - **professional_name**: Nome do profissional que deve constar no certificado, ex: 'João da Silva'.

    Retorna 'Válido' ou 'Inválido' de acordo com a análise do certificado.
    """
    try:
        return await verify_usecase.execute(request)
    except Exception as e:
        logger.error(f"Erro no endpoint verify_file: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed: {e}")
