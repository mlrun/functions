import os
import zipfile
import urllib.request
import tarfile
import json

from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

from typing import Union
import boto3
from urllib.parse import urlparse

def open_archive(
    context: MLClientCtx,
    archive_url: DataItem,
    subdir: str = "content",
    key: str = "content",
    target_path: str = None,
):
    """Open a file/object archive into a target directory
    Currently supports zip and tar.gz
    :param context:      function execution context
    :param archive_url:  url of archive file 
    :param subdir:       path within artifact store where extracted files
                         are stored
    :param key:          key of archive contents in artifact store
    :param target_path:  file system path to store extracted files (use either this or subdir, s3 is valid)
    """

        
    archive_url = archive_url.local()
    
    if (target_path.startswith('s3') or subdir.startswith('s3')):
        if('minio' in os.environ.get('S3_ENDPOINT_URL')):
            client = boto3.client('s3', endpoint_url = os.environ.get('S3_ENDPOINT_URL')) 
        else:
            client = boto3.client('s3')  
            
        if archive_url.endswith("gz"):
            with tarfile.open(archive_url, mode="r|gz") as ref:
                for filename in ref.namelist():
                    data=ref.read(filename)
                    client.put_object(Body=data, Bucket=urlparse(target_path).netloc, Key=f'{urlparse(target_path).path[1:]}/{filename}')

        elif archive_url.endswith("zip"):
            with zipfile.ZipFile(archive_url, "r") as ref:
                for filename in ref.namelist():
                    data=ref.read(filename)
                    client.put_object(Body=data, Bucket=urlparse(target_path).netloc, Key=f'{urlparse(target_path).path[1:]}/{filename}')

        
    else:
        os.makedirs(target_path or subdir, exist_ok=True)

        if archive_url.endswith("gz"):
            with tarfile.open(archive_url, mode="r|gz") as ref:
                ref.extractall(target_path or subdir)
        elif archive_url.endswith("zip"):
            with zipfile.ZipFile(archive_url, "r") as ref:
                ref.extractall(target_path or subdir)
        else:
            raise ValueError(f"unsupported archive type in {archive_url}")

    kwargs = {}
    if target_path:
        kwargs = {"target_path": target_path}
    else:
        kwargs = {"local_path": subdir}
        
    context.log_artifact(key, **kwargs)