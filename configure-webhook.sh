#!/bin/bash
# Configure MinIO webhook to notify worker on file changes

set -e

MINIO_ALIAS="myminio"
MINIO_ENDPOINT="http://minio:9000"
MINIO_USER="minioadmin"
MINIO_PASS="minioadmin"
BUCKET_NAME="rag-documents"

echo "Waiting for MinIO to be ready..."
until mc alias set $MINIO_ALIAS $MINIO_ENDPOINT $MINIO_USER $MINIO_PASS 2>/dev/null; do
    sleep 2
done

echo "Creating bucket..."
mc mb --ignore-existing $MINIO_ALIAS/$BUCKET_NAME

echo "Configuring webhook notifications..."
mc event add $MINIO_ALIAS/$BUCKET_NAME arn:minio::webhook:worker --suffix .pdf --event put,delete
mc event add $MINIO_ALIAS/$BUCKET_NAME arn:minio::webhook:worker --suffix .txt --event put,delete
mc event add $MINIO_ALIAS/$BUCKET_NAME arn:minio::webhook:worker --suffix .docx --event put,delete

echo "Webhook configured successfully!"
