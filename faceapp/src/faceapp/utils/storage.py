import boto3
import os
from botocore.exceptions import ClientError


class S3Utils:
    def __init__(
        self, aws_access_key_id=None, aws_secret_access_key=None, region_name=None
    ):
        """
        Initialize S3 client. If credentials are not provided, boto3 will use the default credentials
        (from environment variables, AWS config file, or IAM role).
        """
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    def download_file(
        self, bucket_name: str, object_key: str, download_dir: str = "./downloads"
    ) -> str:
        """
        Download a file from S3 and return the local path.
        """
        os.makedirs(download_dir, exist_ok=True)
        local_path = os.path.join(download_dir, os.path.basename(object_key))

        try:
            self.s3.download_file(bucket_name, object_key, local_path)
            return local_path
        except ClientError as e:
            print(f"Error downloading {object_key} from {bucket_name}: {e}")
            return ""

    def upload_file(
        self, file_path: str, bucket_name: str, object_key: str = None
    ) -> bool:
        """
        Upload a local file to an S3 bucket.
        """
        if object_key is None:
            object_key = os.path.basename(file_path)

        try:
            self.s3.upload_file(file_path, bucket_name, object_key)
            return True
        except ClientError as e:
            print(f"Error uploading {file_path} to {bucket_name}/{object_key}: {e}")
            return False

    def list_files(self, bucket_name: str, prefix: str = "") -> list:
        """
        List files in an S3 bucket with an optional prefix.
        """
        try:
            response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            return [item["Key"] for item in response.get("Contents", [])]
        except ClientError as e:
            print(f"Error listing files in {bucket_name}/{prefix}: {e}")
            return []

    def delete_file(self, bucket_name: str, object_key: str) -> bool:
        """
        Delete a file from an S3 bucket.
        """
        try:
            self.s3.delete_object(Bucket=bucket_name, Key=object_key)
            return True
        except ClientError as e:
            print(f"Error deleting {object_key} from {bucket_name}: {e}")
            return False

    def move_file(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str = None,
    ) -> bool:
        """
        Move a file from one S3 bucket to another (copy + delete).
        """
        if dest_key is None:
            dest_key = source_key  # keep same filename

        try:
            # Copy
            copy_source = {"Bucket": source_bucket, "Key": source_key}
            self.s3.copy(copy_source, dest_bucket, dest_key)

            # Delete original
            self.s3.delete_object(Bucket=source_bucket, Key=source_key)
            return True
        except ClientError as e:
            print(
                f"Error moving {source_key} from {source_bucket} to {dest_bucket}/{dest_key}: {e}"
            )
            return False
