# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
# ******************************************************************************

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import io
import posixpath
import zipfile
from os import walk, path

import requests
from tqdm import tqdm


def download_file(url, sourcefile, destfile, totalsz=None):
    """
    Download the file specified by the given URL.

    Args:
        url (str): url to download from
        sourcefile (str): file to download from url
        destfile (str): save path
        totalsz (:obj:`int`, optional): total size of file
    """
    req = requests.get(posixpath.join(url, sourcefile),
                       stream=True)

    chunksz = 1024 ** 2
    if totalsz is None:
        if "Content-length" in req.headers:
            totalsz = int(req.headers["Content-length"])
            nchunks = totalsz // chunksz
        else:
            print("Unable to determine total file size.")
            nchunks = None
    else:
        nchunks = totalsz // chunksz

    print("Downloading file to: {}".format(destfile))
    with open(destfile, 'wb') as f:
        for data in tqdm(req.iter_content(chunksz), total=nchunks, unit="MB"):
            f.write(data)
    print("Download Complete")


def unzip_file(filepath, outpath='.'):
    """
    Unzip a file to the same location of filepath

    Args:
        filepath (str): path to file
        outpath (str): path to extract to

    """
    z = zipfile.ZipFile(filepath, 'r')
    z.extractall(outpath)
    z.close()
