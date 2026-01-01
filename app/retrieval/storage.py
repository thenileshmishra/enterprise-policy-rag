"""Optional S3-backed storage helpers for FAISS index persistence."""
import os
from botocore.exceptions import ClientError
import boto3
from app.core.logger import logger


def s3_client():
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    endpoint = os.getenv("S3_ENDPOINT")  # e.g., https://nyc3.digitaloceanspaces.com

    session = boto3.session.Session()
    if endpoint:
        return session.client(
            "s3",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=region,
            endpoint_url=endpoint,
        )

    return session.client(
        "s3",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=region,
    )


def download_index(bucket: str, index_key: str, meta_key: str, dest_index: str, dest_meta: str):
    client = s3_client()
    try:
        logger.info(f"Attempting to download FAISS index from s3://{bucket}/{index_key}")
        client.download_file(bucket, index_key, dest_index)
        client.download_file(bucket, meta_key, dest_meta)
        logger.info("FAISS index download successful")
        return True
    except ClientError as e:
        logger.warning(f"Could not download FAISS index: {e}")
        return False


def upload_index(bucket: str, index_key: str, meta_key: str, src_index: str, src_meta: str):
    client = s3_client()
    try:
        logger.info(f"Uploading FAISS index to s3://{bucket}/{index_key}")
        client.upload_file(src_index, bucket, index_key)
        client.upload_file(src_meta, bucket, meta_key)
        logger.info("Upload successful")
        return True
    except ClientError:
        logger.exception("Failed to upload index to S3")
        return False
