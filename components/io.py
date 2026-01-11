import os
from io import BytesIO
import pandas as pd
import boto3


def s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION"),
    )


def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return pd.read_parquet(BytesIO(data))


def read_csv_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])
