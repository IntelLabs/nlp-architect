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

import io
import os
import posixpath
import zipfile
from os import walk, path
from pathlib import Path

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


def walk_directory(directory):
    """Iterates a directory's text files and their contents."""
    for dir_path, _, filenames in walk(directory):
        for filename in filenames:
            file_path = path.join(dir_path, filename)
            if path.isfile(file_path) and not filename.startswith('.'):
                with io.open(file_path, 'r', encoding='utf-8') as file:
                    print('Reading ' + filename)
                    doc_text = file.read()
                    yield filename, doc_text


def validate(*args):
    """
    Validate all arguments are of correct type and in correct range.
    Args:
        *args (tuple of tuples): Each tuple represents an argument validation like so:
        Option 1 - With range check:
            (arg, class, min_val, max_val)
        Option 2 - Without range check:
            (arg, class)
        If class is a tuple of type objects check if arg is an instance of any of the types.
        To allow a None valued argument, include type(None) in class.
        To disable lower or upper bound check, set min_val or max_val to None, respectively.
        If arg has the len attribute (such as string), range checks are performed on its length.
    """
    for arg in args:
        if not isinstance(arg[0], arg[1]):
            raise TypeError
        if arg[0] and len(arg) == 4:
            num = len(arg[0]) if hasattr(arg[0], '__len__') else arg[0]
            if arg[2] and num < arg[2] or arg[3] and num > arg[3]:
                raise ValueError


def validate_existing_filepath(arg):
    """Validates an input argument is a path string to an existing file, and returns it."""
    validate((arg, str, 0, 255))
    if not os.path.isfile(arg):
        raise ValueError("{0} is not a path to existing file.".format(arg))
    return arg


def validate_existing_directory(arg):
    """Validates an input argument is a path string to an existing directory, and returns it."""
    validate((arg, str, 0, 255))
    if not os.path.isdir(arg):
        raise ValueError("{0} is not a path to existing directory".format(arg))
    return arg


def sanitize_path(path):
    s_path = os.path.normpath('/' + path).lstrip('/')
    assert len(s_path) < 255
    return s_path
