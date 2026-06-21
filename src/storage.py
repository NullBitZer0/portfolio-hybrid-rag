import os
from minio import Minio
from minio.error import S3Error
from src.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, MINIO_SECURE


class MinIOStorage:
    """MinIO client for document storage."""

    def __init__(self):
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        self.bucket = MINIO_BUCKET
        try:
            self._ensure_bucket()
        except Exception as e:
            print(f"Warning: MinIO not ready ({e}). Will retry on first use.")

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            print(f"Created MinIO bucket: {self.bucket}")

    def _retry_ensure_bucket(self):
        """Retry bucket creation."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                print(f"Created MinIO bucket: {self.bucket}")
        except Exception:
            pass

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """Upload bytes to MinIO."""
        import io
        self._retry_ensure_bucket()
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        print(f"Uploaded bytes to {self.bucket}/{object_name}")
        return object_name

    def download_file(self, object_name: str, file_path: str) -> str:
        """Download a file from MinIO."""
        self.client.fget_object(self.bucket, object_name, file_path)
        print(f"Downloaded {self.bucket}/{object_name} to {file_path}")
        return file_path

    def list_files(self, prefix: str = "", recursive: bool = True) -> list[dict]:
        """List files in MinIO bucket."""
        objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=recursive)
        return [
            {
                "name": obj.object_name,
                "size": obj.size,
                "last_modified": obj.last_modified,
            }
            for obj in objects
        ]

    def delete_file(self, object_name: str) -> bool:
        """Delete a file from MinIO."""
        try:
            self.client.remove_object(self.bucket, object_name)
            print(f"Deleted {self.bucket}/{object_name}")
            return True
        except S3Error as e:
            print(f"Error deleting {object_name}: {e}")
            return False

    def file_exists(self, object_name: str) -> bool:
        """Check if file exists in MinIO."""
        try:
            self.client.stat_object(self.bucket, object_name)
            return True
        except S3Error:
            return False


storage = MinIOStorage()
