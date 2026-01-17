"""
AWS S3 storage helpers for FAISS index and PDF persistence.
Supports both AWS S3 and S3-compatible services (DigitalOcean Spaces, MinIO, etc.).
"""

import os
import io
from typing import Optional, Tuple, BinaryIO
from pathlib import Path
from datetime import datetime
from botocore.exceptions import ClientError
import boto3
from app.core.logger import logger


def get_s3_client():
    """
    Create S3 client with configuration from environment variables.
    Supports both AWS S3 and S3-compatible endpoints.
    """
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    endpoint = os.getenv("S3_ENDPOINT")  # For S3-compatible services

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


# Alias for backward compatibility
s3_client = get_s3_client


def download_index(
    bucket: str,
    index_key: str,
    meta_key: str,
    dest_index: str,
    dest_meta: str
) -> bool:
    """
    Download FAISS index and metadata from S3.

    Args:
        bucket: S3 bucket name
        index_key: S3 key for the FAISS index file
        meta_key: S3 key for the metadata JSON file
        dest_index: Local destination path for index
        dest_meta: Local destination path for metadata

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()
    try:
        logger.info(f"Downloading FAISS index from s3://{bucket}/{index_key}")
        client.download_file(bucket, index_key, dest_index)
        client.download_file(bucket, meta_key, dest_meta)
        logger.info("FAISS index download successful")
        return True
    except ClientError as e:
        logger.warning(f"Could not download FAISS index: {e}")
        return False


def upload_index(
    bucket: str,
    index_key: str,
    meta_key: str,
    src_index: str,
    src_meta: str
) -> bool:
    """
    Upload FAISS index and metadata to S3.

    Args:
        bucket: S3 bucket name
        index_key: S3 key for the FAISS index file
        meta_key: S3 key for the metadata JSON file
        src_index: Local source path for index
        src_meta: Local source path for metadata

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()
    try:
        logger.info(f"Uploading FAISS index to s3://{bucket}/{index_key}")
        client.upload_file(src_index, bucket, index_key)
        client.upload_file(src_meta, bucket, meta_key)
        logger.info("Upload successful")
        return True
    except ClientError:
        logger.exception("Failed to upload index to S3")
        return False


def upload_pdf(
    file_path: str,
    bucket: str,
    session_id: Optional[str] = None,
    custom_key: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Upload a PDF file to S3.

    Args:
        file_path: Local path to the PDF file
        bucket: S3 bucket name
        session_id: Optional session ID for organizing uploads
        custom_key: Optional custom S3 key (overrides default naming)

    Returns:
        Tuple of (success, s3_key)
    """
    client = get_s3_client()
    path = Path(file_path)

    if not path.exists():
        logger.error(f"PDF file not found: {file_path}")
        return False, None

    # Generate S3 key
    if custom_key:
        s3_key = custom_key
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if session_id:
            s3_key = f"pdfs/{session_id}/{timestamp}_{path.name}"
        else:
            s3_key = f"pdfs/{timestamp}_{path.name}"

    try:
        logger.info(f"Uploading PDF to s3://{bucket}/{s3_key}")
        client.upload_file(
            file_path,
            bucket,
            s3_key,
            ExtraArgs={"ContentType": "application/pdf"}
        )
        logger.info(f"PDF upload successful: {s3_key}")
        return True, s3_key
    except ClientError as e:
        logger.exception(f"Failed to upload PDF to S3: {e}")
        return False, None


def upload_pdf_from_bytes(
    file_bytes: bytes,
    filename: str,
    bucket: str,
    session_id: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Upload PDF from bytes directly to S3 (for file uploads).

    Args:
        file_bytes: PDF file content as bytes
        filename: Original filename
        bucket: S3 bucket name
        session_id: Optional session ID for organizing uploads

    Returns:
        Tuple of (success, s3_key)
    """
    client = get_s3_client()

    # Generate S3 key
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if session_id:
        s3_key = f"pdfs/{session_id}/{timestamp}_{filename}"
    else:
        s3_key = f"pdfs/{timestamp}_{filename}"

    try:
        logger.info(f"Uploading PDF bytes to s3://{bucket}/{s3_key}")
        client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=file_bytes,
            ContentType="application/pdf"
        )
        logger.info(f"PDF upload successful: {s3_key}")
        return True, s3_key
    except ClientError as e:
        logger.exception(f"Failed to upload PDF bytes to S3: {e}")
        return False, None


def download_pdf(
    bucket: str,
    s3_key: str,
    dest_path: str
) -> bool:
    """
    Download a PDF from S3.

    Args:
        bucket: S3 bucket name
        s3_key: S3 key for the PDF
        dest_path: Local destination path

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()
    try:
        logger.info(f"Downloading PDF from s3://{bucket}/{s3_key}")
        client.download_file(bucket, s3_key, dest_path)
        logger.info("PDF download successful")
        return True
    except ClientError as e:
        logger.warning(f"Could not download PDF: {e}")
        return False


def get_pdf_url(
    bucket: str,
    s3_key: str,
    expiration: int = 3600
) -> Optional[str]:
    """
    Generate a presigned URL for PDF access.

    Args:
        bucket: S3 bucket name
        s3_key: S3 key for the PDF
        expiration: URL expiration time in seconds (default 1 hour)

    Returns:
        Presigned URL or None if failed
    """
    client = get_s3_client()
    try:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": s3_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        return None


def list_session_pdfs(bucket: str, session_id: str) -> list:
    """
    List all PDFs uploaded in a session.

    Args:
        bucket: S3 bucket name
        session_id: Session ID

    Returns:
        List of S3 keys for PDFs in the session
    """
    client = get_s3_client()
    prefix = f"pdfs/{session_id}/"

    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [obj["Key"] for obj in response.get("Contents", [])]
        logger.info(f"Found {len(keys)} PDFs in session {session_id}")
        return keys
    except ClientError as e:
        logger.error(f"Failed to list session PDFs: {e}")
        return []


def delete_session_pdfs(bucket: str, session_id: str) -> bool:
    """
    Delete all PDFs for a session (cleanup).

    Args:
        bucket: S3 bucket name
        session_id: Session ID

    Returns:
        True if successful, False otherwise
    """
    client = get_s3_client()
    keys = list_session_pdfs(bucket, session_id)

    if not keys:
        logger.info(f"No PDFs to delete for session {session_id}")
        return True

    try:
        objects = [{"Key": key} for key in keys]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects}
        )
        logger.info(f"Deleted {len(keys)} PDFs from session {session_id}")
        return True
    except ClientError as e:
        logger.error(f"Failed to delete session PDFs: {e}")
        return False


def check_s3_connection(bucket: str) -> bool:
    """
    Check if S3 connection is working.

    Args:
        bucket: S3 bucket name to check

    Returns:
        True if connection successful, False otherwise
    """
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=bucket)
        logger.info(f"S3 connection verified for bucket: {bucket}")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        logger.error(f"S3 connection failed: {error_code}")
        return False
