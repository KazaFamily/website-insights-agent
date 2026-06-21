"""
Reads data files from S3 into pandas DataFrames.
Supports GA4 exports and Search Console CSVs.
"""

import io
import boto3
import pandas as pd


def _s3_client():
    # Uses AWS credentials from environment / IAM role
    return boto3.client("s3")


def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Download a single CSV from S3 and return as a DataFrame."""
    client = _s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(response["Body"].read()))


def list_files_in_prefix(bucket: str, prefix: str) -> list[str]:
    """Return all object keys under a given S3 prefix."""
    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def load_ga4_pages_data(bucket: str, prefix: str) -> pd.DataFrame:
    """
    Load and concatenate all GA4 CSV exports under the given S3 prefix.
    Returns an empty DataFrame if no files are found.
    """
    keys = list_files_in_prefix(bucket, prefix)
    csv_keys = [k for k in keys if k.endswith(".csv")]
    if not csv_keys:
        return pd.DataFrame()
    frames = [read_csv_from_s3(bucket, key) for key in csv_keys]
    return pd.concat(frames, ignore_index=True)

def load_ga4_traffic_data(bucket: str, prefix: str) -> pd.DataFrame:
    """
    Load and concatenate all GA4 CSV exports under the given S3 prefix.
    Returns an empty DataFrame if no files are found.
    """
    keys = list_files_in_prefix(bucket, prefix)
    csv_keys = [k for k in keys if k.endswith(".csv")]
    if not csv_keys:
        return pd.DataFrame()
    frames = [read_csv_from_s3(bucket, key) for key in csv_keys]
    return pd.concat(frames, ignore_index=True)

def load_search_console_pages_data(bucket: str, prefix: str) -> pd.DataFrame:
    """
    Load and concatenate all Search Console CSV exports under the given S3 prefix.
    Returns an empty DataFrame if no files are found.
    """
    keys = list_files_in_prefix(bucket, prefix)
    csv_keys = [k for k in keys if k.endswith(".csv")]
    if not csv_keys:
        return pd.DataFrame()
    frames = [read_csv_from_s3(bucket, key) for key in csv_keys]
    return pd.concat(frames, ignore_index=True)

def load_search_console_queries_data(bucket: str, prefix: str) -> pd.DataFrame:
    """
    Load and concatenate all Search Console CSV exports under the given S3 prefix.
    Returns an empty DataFrame if no files are found.
    """
    keys = list_files_in_prefix(bucket, prefix)
    csv_keys = [k for k in keys if k.endswith(".csv")]
    if not csv_keys:
        return pd.DataFrame()
    frames = [read_csv_from_s3(bucket, key) for key in csv_keys]
    return pd.concat(frames, ignore_index=True)
