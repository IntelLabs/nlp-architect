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
import argparse
import gzip
import io
import json
import os
import posixpath
import re
import sys
import zipfile
from os import PathLike, makedirs, remove
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from nlp_architect.utils.generic import license_prompt


def download_unlicensed_file(url, sourcefile, destfile, totalsz=None):
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
        for data in tqdm(req.iter_content(chunksz), total=nchunks, unit="MB", file=sys.stdout):
            f.write(data)
    print("Download Complete")


def uncompress_file(filepath: str or os.PathLike, outpath='.'):
    """
    Unzip a file to the same location of filepath
    uses decompressing algorithm by file extension

    Args:
        filepath (str): path to file
        outpath (str): path to extract to
    """
    filepath = str(filepath)
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath) as z:
            z.extractall(outpath)
    elif filepath.endswith('.gz'):
        if os.path.isdir(outpath):
            raise ValueError('output path for gzip must be a file')
        with gzip.open(filepath, 'rb') as fp:
            file_content = fp.read()
        with open(outpath, 'wb') as fp:
            fp.write(file_content)
    else:
        raise ValueError('Unsupported archive provided. Method supports only .zip/.gz files.')


def gzip_str(g_str):
    """
    Transform string to GZIP coding

    Args:
        g_str (str): string of data

    Returns:
        GZIP bytes data
    """
    compressed_str = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_str, mode='w') as file_out:
        file_out.write((json.dumps(g_str).encode()))
    bytes_obj = compressed_str.getvalue()
    return bytes_obj


def check_directory_and_create(dir_path):
    """
    Check if given directory exists, create if not.

    Args:
        dir_path (str): path to directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def walk_directory(directory, verbose=False):
    """Iterates a directory's text files and their contents."""
    for dir_path, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                with io.open(file_path, 'r', encoding='utf-8') as file:
                    if verbose:
                        print('Reading {}'.format(filename))
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
        arg_val = arg[0]
        arg_type = (arg[1],) if isinstance(arg[1], type) else arg[1]
        if not isinstance(arg_val, arg_type):
            raise TypeError('Expected type {}'.format(' or '.join([t.__name__ for t in arg_type])))
        if arg_val is not None and len(arg) >= 4:
            name = 'of ' + arg[4] if len(arg) == 5 else ''
            arg_min = arg[2]
            arg_max = arg[3]
            if hasattr(arg_val, '__len__'):
                val = 'Length'
                num = len(arg_val)
            else:
                val = 'Value'
                num = arg_val
            if arg_min is not None and num < arg_min:
                raise ValueError('{} {} must be greater or equal to {}'.format(val, name, arg_min))
            if arg_max is not None and num >= arg_max:
                raise ValueError('{} {} must be less than {}'.format(val, name, arg_max))


def validate_existing_filepath(arg):
    """Validates an input argument is a path string to an existing file."""
    validate((arg, str, 0, 255))
    if not os.path.isfile(arg):
        raise ValueError("{0} does not exist.".format(arg))
    return arg


def validate_existing_directory(arg):
    """Validates an input argument is a path string to an existing directory."""
    arg = os.path.abspath(arg)
    validate((arg, str, 0, 255))
    if not os.path.isdir(arg):
        raise ValueError("{0} does not exist".format(arg))
    return arg


def validate_existing_path(arg):
    """Validates an input argument is a path string to an existing file or directory."""
    arg = os.path.abspath(arg)
    validate((arg, str, 0, 255))
    if not os.path.exists(arg):
        raise ValueError("{0} does not exist".format(arg))
    return arg


def validate_parent_exists(arg):
    """Validates an input argument is a path string, and its parent directory exists."""
    arg = os.path.abspath(arg)
    dir_arg = os.path.dirname(os.path.abspath(arg))
    if validate_existing_directory(dir_arg):
        return arg
    return None


def valid_path_append(path, *args):
    """
    Helper to validate passed path directory and append any subsequent
    filename arguments.

    Arguments:
        path (str): Initial filesystem path.  Should expand to a valid
                    directory.
        *args (list, optional): Any filename or path suffices to append to path
                                for returning.
        Returns:
            (list, str): path prepended list of files from args, or path alone if
                     no args specified.
    Raises:
        ValueError: if path is not a valid directory on this filesystem.
    """
    full_path = os.path.expanduser(path)
    res = []
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    if not os.path.isdir(full_path):
        raise ValueError("path: {0} is not a valid directory".format(path))
    for suffix_path in args:
        res.append(os.path.join(full_path, suffix_path))
    if len(res) == 0:
        return path
    if len(res) == 1:
        return res[0]
    return res


def sanitize_path(path):
    s_path = os.path.normpath('/' + path).lstrip('/')
    assert len(s_path) < 255
    return s_path


def check(validator):
    class CustomAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            validator(values)
            setattr(namespace, self.dest, values)

    return CustomAction


def check_size(min_size=None, max_size=None):
    class CustomAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            validate((values, self.type, min_size, max_size, self.dest))
            setattr(namespace, self.dest, values)

    return CustomAction


def validate_proxy_path(arg):
    """Validates an input argument is a valid proxy path or None"""
    proxy_validation_regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    if arg is not None and re.match(proxy_validation_regex, arg) is None:
        raise ValueError("{0} is not a valid proxy path".format(arg))
    return arg


def validate_boolean(arg):
    """Validates an input argument of type boolean"""
    if arg.lower() not in ['true', 'false']:
        raise argparse.ArgumentTypeError('expected true | false argument')
    return arg.lower() == "true"


def load_json_file(file_path):
    """load a file into a json object"""
    try:
        with open(file_path) as small_file:
            return json.load(small_file)
    except OSError as e:
        print(e)
        print('trying to read file in blocks')
        with open(file_path) as big_file:
            json_string = ''
            while True:
                block = big_file.read(64 * (1 << 20))  # Read 64 MB at a time;
                json_string = json_string + block
                if not block:  # Reached EOF
                    break
            return json.loads(json_string)


def json_dumper(obj):
    """for objects that have members that cant be serialized and implement toJson() method"""
    try:
        return obj.toJson()
    except Exception:
        return obj.__dict__


def load_files_from_path(dir_path, extension='txt'):
    """load all files from given directory (with given extension)"""
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(extension)]
    files_data = []
    for f in files:
        with open(f) as fp:
            files_data.append(' '.join(map(str.strip, fp.readlines())))
    return files_data


def create_folder(path):
    if path:
        if not os.path.exists(path):
            os.makedirs(path)


def download_unzip(url: str, sourcefile: str, unzipped_path: str or PathLike,
                   license_msg: str = None):
    """Downloads a zip file, extracts it to destination, deletes the zip file. If license_msg is
    supplied, user is prompted for download confirmation."""
    dest_parent = Path(unzipped_path).parent

    if not os.path.exists(unzipped_path):
        if license_msg is None or license_prompt(license_msg, urlparse(url).netloc):
            zip_path = dest_parent / sourcefile
            makedirs(dest_parent, exist_ok=True)
            download_unlicensed_file(url, sourcefile, zip_path)
            print('Unzipping...')
            uncompress_file(zip_path, dest_parent)
            remove(zip_path)
    return unzipped_path
