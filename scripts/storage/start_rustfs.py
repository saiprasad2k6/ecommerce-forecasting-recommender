#!/usr/bin/env python3
"""Start local RustFS, create the DVC bucket and configure private credentials."""
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from botocore.config import Config
from botocore.exceptions import (
    ClientError, ConnectionClosedError, ConnectTimeoutError,
    EndpointConnectionError, ReadTimeoutError,
)
from botocore.session import get_session
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
STORAGE = ROOT / "storage"


def wait_for_storage(client, attempts=120):
    """Wait for S3 readiness, including temporary HTTP errors during startup."""
    for attempt in range(attempts):
        try:
            client.list_buckets()
            return
        except (ClientError, EndpointConnectionError, ConnectionClosedError,
                ConnectTimeoutError, ReadTimeoutError) as error:
            if isinstance(error, ClientError):
                status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status not in {429, 500, 502, 503, 504}:
                    raise
            if attempt == attempts - 1:
                raise RuntimeError(
                    "RustFS did not become ready; inspect the rustfs container logs."
                ) from error
            time.sleep(1)


def main():
    STORAGE.mkdir(parents=True, exist_ok=True)
    credentials = STORAGE / ".env"
    if not credentials.exists():
        with open(credentials, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
            stream.write(f"RUSTFS_ACCESS_KEY=olist-{secrets.token_hex(6)}\n")
            stream.write(f"RUSTFS_SECRET_KEY={secrets.token_urlsafe(32)}\n")
    values = dotenv_values(credentials)
    endpoint = f"http://127.0.0.1:{values.get('RUSTFS_API_PORT', '19000')}"
    console = f"http://127.0.0.1:{values.get('RUSTFS_CONSOLE_PORT', '19001')}"
    subprocess.run([
        "docker", "compose", "--env-file", str(credentials),
        "-f", str(STORAGE / "compose.yaml"), "up", "-d",
    ], check=True)

    client = get_session().create_client(
        "s3", endpoint_url=endpoint, region_name="us-east-1",
        aws_access_key_id=values["RUSTFS_ACCESS_KEY"],
        aws_secret_access_key=values["RUSTFS_SECRET_KEY"],
        config=Config(connect_timeout=1, read_timeout=2, retries={"max_attempts": 0},
                      s3={"addressing_style": "path"}),
    )
    wait_for_storage(client)
    try:
        client.head_bucket(Bucket="olist-dvc")
    except ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
            raise
        client.create_bucket(Bucket="olist-dvc")

    dvc = str(Path(sys.executable).with_name("dvc"))
    subprocess.run([dvc, "remote", "modify", "--local", "rustfs", "endpointurl", endpoint], cwd=ROOT, check=True)
    for key, value in {
        "access_key_id": values["RUSTFS_ACCESS_KEY"],
        "secret_access_key": values["RUSTFS_SECRET_KEY"],
    }.items():
        subprocess.run([dvc, "remote", "modify", "--local", "rustfs", key, value], cwd=ROOT, check=True)
    (ROOT / ".dvc/config.local").chmod(0o600)
    print(f"RustFS console: {console}")
    print(f"DVC remote: s3://olist-dvc/cache ({endpoint})")
    print(f"Login credentials: {credentials} (not committed to Git)")


if __name__ == "__main__":
    main()
