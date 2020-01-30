# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import zipfile
import urllib
import tarfile
import json
from tempfile import mktemp

def open_archive(context, 
                 target_dir='content',
                 archive_url=''):
    """Open a file/object archive into a target directory
    
    Currently supports zip and tar.gz
    """
        
    # Define locations
    os.makedirs(target_dir, exist_ok=True)
    context.logger.info('Verified directories')
    
    splits = archive_url.split('.')
    if ('.'.join(splits[-2:]) == 'tar.gz'):
        # Extract dataset from tar
        context.logger.info('opening tar_gz')
        ftpstream = urllib.request.urlopen(archive_url)
        ref = tarfile.open(fileobj=ftpstream, mode="r|gz")
    elif splits[-1] == 'zip':
        # Extract dataset from zip
        context.logger.info('opening zip')
        ref = zipfile.ZipFile(archive_url, 'r')

    ref.extractall(target_dir)
    ref.close()
    
    context.logger.info(f'extracted archive to {target_dir}')
    context.log_artifact('content', target_path=target_dir)
