from azure.core.exceptions import ResourceExistsError
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from config.logs import logger
from config.settings import settings


class Storage:
    async def get_blob_service_client(self) -> BlobServiceClient:
        logger.info("Obtendo BlobServiceClient.")
        return BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )

    async def create_container(self, container_name: str) -> ContainerClient:
        blob_service_client = await self.get_blob_service_client()
        try:
            container_client = await blob_service_client.create_container(
                container_name
            )
            logger.info(f"Container '{container_name}' criado com sucesso.")
        except ResourceExistsError:
            container_client = blob_service_client.get_container_client(container_name)
            logger.info(f"Container '{container_name}' já existe.")
        return container_client

    async def upload_file(
        self, container_name: str, filename: str, content: bytes
    ) -> str:
        logger.info(
            f"Iniciando upload do arquivo '{filename}' para o container '{container_name}'."
        )
        container_client = await self.create_container(container_name)
        blob_client = container_client.get_blob_client(filename)
        try:
            await blob_client.upload_blob(content, overwrite=True)
            logger.info(f"Upload do arquivo '{filename}' concluído com sucesso.")
            return f"{container_name}/{filename}"
        except Exception as e:
            logger.error(f"Erro ao realizar upload do arquivo '{filename}': {e}")
            raise

    async def download_file(self, blob_path: str) -> bytes | None:
        logger.info(f"Iniciando download do blob '{blob_path}'.")
        try:
            container_name, blob_name = blob_path.split("/", 1)
            blob_service_client = await self.get_blob_service_client()
            async with blob_service_client:
                container_client = blob_service_client.get_container_client(
                    container_name
                )
                blob_client = container_client.get_blob_client(blob_name)
                download_stream = await blob_client.download_blob()
                content = await download_stream.readall()
                logger.info(f"Download do blob '{blob_path}' concluído com sucesso.")
                return content
        except Exception as e:
            logger.error(f"Erro ao realizar download do blob '{blob_path}': {e}")
            return None
