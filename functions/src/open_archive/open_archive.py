# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import tarfile
import zipfile
from urllib.parse import urlparse

from mlrun.artifacts.base import DirArtifact
from mlrun.datastore import DataItem
from mlrun.execution import MLClientCtx


def open_archive(
    context: MLClientCtx,
    archive_url: DataItem,
    subdir: str = "content/",
    key: str = "content",
    target_path: str = None,
):
    """Open a file/object archive into a target directory. Currently, supports zip and tar.gz.

    :param context:      function execution context
    :param archive_url:  url of archive file
    :param subdir:       path within artifact store where extracted files are stored, default is "/content"
    :param key:          key of archive contents in artifact store
    :param target_path:  file system path to store extracted files
    """

    # Resolves the archive locally
    archive_url = archive_url.local()
    v3io_subdir = None
    # When custom artifact path is defined
    if not target_path and context.artifact_path:
        parsed_subdir = urlparse(context.artifact_path)
        if parsed_subdir.scheme == "s3":
            subdir = os.path.join(context.artifact_path, subdir)
        elif parsed_subdir.scheme == "v3io":
            v3io_subdir = os.path.join(
                context.artifact_path, subdir
            )  # Using v3io_subdir for logging
            subdir = "/v3io" + parsed_subdir.path + "/" + subdir
            context.logger.info(f"Using v3io scheme, extracting to {subdir}")
        else:
            context.logger.info(f"Unrecognizable scheme, extracting to {subdir}")

    # When working on CE, target path might be on s3
    if "s3" in (target_path or subdir):
        context.logger.info(f"Using s3 scheme, extracting to {target_path or subdir}")

        if archive_url.endswith("gz"):
            _extract_gz_file(
                archive_url=archive_url,
                subdir=subdir,
                target_path=target_path,
                in_s3=True,
            )

        elif archive_url.endswith("zip"):
            _extract_zip_file(
                archive_url=archive_url,
                subdir=subdir,
                target_path=target_path,
                in_s3=True,
            )
        else:
            raise ValueError(f"unsupported archive type in {archive_url}")
    else:
        if archive_url.endswith("gz"):
            _extract_gz_file(
                archive_url=archive_url, subdir=subdir, target_path=target_path
            )
        elif archive_url.endswith("zip"):
            _extract_zip_file(
                archive_url=archive_url, subdir=subdir, target_path=target_path
            )
        else:
            raise ValueError(f"unsupported archive type in {archive_url}")

    if v3io_subdir:
        subdir = v3io_subdir

    context.logger.info(f"Logging artifact to {(target_path or subdir)}")
    context.log_artifact(DirArtifact(key=key, target_path=(target_path or subdir)))


def _extract_gz_file(
    archive_url: str,
    target_path: str = None,
    subdir: str = "content/",
    in_s3: bool = False,
):
    if in_s3:
        client = _init_boto3_client()
        with tarfile.open(archive_url, mode="r|gz") as ref:
            for member in ref.getmembers():
                data = ref.extractfile(member=member).read()
                client.put_object(
                    Body=data,
                    Bucket=urlparse(target_path or subdir).netloc,
                    Key=f"{urlparse(target_path or subdir).path[1:]}{member.name}",
                )
    else:
        os.makedirs(target_path or subdir, exist_ok=True)
        with tarfile.open(archive_url, mode="r:gz") as ref:
            for entry in ref:
                # Validate that there is no path traversal in the archive
                if os.path.isabs(entry.name) or ".." in entry.name:
                    raise ValueError(f"Illegal tar archive entry: {entry.name}")

                ref.extract(entry, target_path or subdir)


def _extract_zip_file(
    archive_url, target_path: str = None, subdir: str = "content/", in_s3: bool = False
):
    if in_s3:
        client = _init_boto3_client()
        with zipfile.ZipFile(archive_url, "r") as ref:
            for filename in ref.namelist():
                data = ref.read(filename)
                client.put_object(
                    Body=data,
                    Bucket=urlparse(target_path or subdir).netloc,
                    Key=f"{urlparse(target_path or subdir).path[1:]}{filename}",
                )
    else:
        with zipfile.ZipFile(archive_url, "r") as ref:
            # Validate that there is no path traversal in the archive
            for entry in ref.namelist():
                if os.path.isabs(entry) or ".." in entry:
                    raise ValueError(f"Illegal zip archive entry: {entry}")
            os.makedirs(target_path or subdir, exist_ok=True)
            ref.extractall(target_path or subdir)


def _init_boto3_client():
    import boto3

    # Backward compatibility: Support both S3_ENDPOINT_URL (deprecated) and AWS_ENDPOINT_URL_S3
    # TODO: Remove this in 1.12.0
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get(
        "S3_ENDPOINT_URL"
    )

    if endpoint_url:
        client = boto3.client("s3", endpoint_url=endpoint_url)
    else:
        client = boto3.client("s3")
    return client
