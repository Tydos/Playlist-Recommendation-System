import os
from pathlib import Path
from dotenv import load_dotenv
import boto3
from botocore.client import Config
from backend.utils.logging import get_logger

load_dotenv()
logger = get_logger("blob")

_client = None


def _is_configured() -> bool:
    return all(os.getenv(k) for k in (
        "SUPABASE_S3_ENDPOINT",
        "SUPABASE_S3_ACCESS_KEY_ID",
        "SUPABASE_S3_SECRET_ACCESS_KEY",
        "SUPABASE_S3_BUCKET",
    ))


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=os.environ["SUPABASE_S3_ENDPOINT"],
            aws_access_key_id=os.environ["SUPABASE_S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["SUPABASE_S3_SECRET_ACCESS_KEY"],
            region_name=os.getenv("SUPABASE_S3_REGION", "us-east-1"),
            config=Config(signature_version="s3v4"),
        )
    return _client


def upload_file(local_path: str, blob_key: str) -> str | None:
    """Upload a single file to Supabase S3 and return its URL. No-ops if S3 is not configured."""
    if not _is_configured():
        logger.warning(f"S3 not configured — skipping upload of {local_path}")
        return None
    bucket = os.environ["SUPABASE_S3_BUCKET"]
    client = _get_client()
    client.upload_file(local_path, bucket, blob_key)
    endpoint = os.environ["SUPABASE_S3_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/{bucket}/{blob_key}"
    logger.info(f"Uploaded {local_path} → {url}")
    return url


def upload_dir(local_dir: str, blob_prefix: str) -> list[str]:
    """Recursively upload all files in a directory, preserving structure."""
    if not _is_configured():
        logger.warning(f"S3 not configured — skipping upload of {local_dir}/")
        return []
    urls = []
    for path in Path(local_dir).rglob("*"):
        if path.is_file():
            relative = path.relative_to(local_dir)
            key = f"{blob_prefix}/{relative}".replace("\\", "/")
            url = upload_file(str(path), key)
            if url:
                urls.append(url)
    return urls
