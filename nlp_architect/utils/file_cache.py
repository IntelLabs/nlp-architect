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
"""
Utilities for working with the local dataset cache.
"""
import os
import logging
import shutil
import tempfile
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Tuple, Union, IO
from hashlib import sha256

from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.io import load_json_file

import requests

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MODEL_CACHE = LIBRARY_OUT / "pretrained_models"


def cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = MODEL_CACHE
    else:
        cache_dir = cache_dir
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    if os.path.exists(url_or_filename):
        # File, and it exists.
        print("File already exists. No further processing needed.")
        return url_or_filename
    if parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))

    # Something unknown
    raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def url_to_filename(url: str, etag: str = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    if url.split("/")[-1].endswith("zip"):
        url_bytes = url.encode("utf-8")
        url_hash = sha256(url_bytes)
        filename = url_hash.hexdigest()
        if etag:
            etag_bytes = etag.encode("utf-8")
            etag_hash = sha256(etag_bytes)
            filename += "." + etag_hash.hexdigest()
    else:
        filename = url.split("/")[-1]

    return filename


def filename_to_url(filename: str, cache_dir: str = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = MODEL_CACHE

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def http_get(url: str, temp_file: IO) -> None:
    req = requests.get(url, stream=True)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            temp_file.write(chunk)


def get_from_cache(url: str, cache_dir: str = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = MODEL_CACHE

    os.makedirs(cache_dir, exist_ok=True)

    response = requests.head(url, allow_redirects=True)
    if response.status_code != 200:
        raise IOError(
            "HEAD request failed for url {} with status code {}".format(url, response.status_code)
        )
    etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    need_downloading = True

    if os.path.exists(cache_path):
        # check if etag has changed comparing with the metadata
        if url.split("/")[-1].endswith("zip"):
            meta_path = cache_path + ".json"
        else:
            meta_path = cache_path + "_meta_" + ".json"
        meta = load_json_file(meta_path)
        if meta["etag"] == etag:
            print("file already present")
            need_downloading = False

    if need_downloading:
        print("File not present or etag changed")
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            if url.split("/")[-1].endswith("zip"):
                meta_path = cache_path + ".json"
            else:
                meta_path = cache_path + "_meta_" + ".json"
            with open(meta_path, "w") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path, need_downloading
