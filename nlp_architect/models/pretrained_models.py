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

from nlp_architect.utils.io import uncompress_file, zipfile_list
from nlp_architect.utils.file_cache import cached_path

from nlp_architect import LIBRARY_OUT

S3_PREFIX = "https://s3-us-west-2.amazonaws.com/nlp-architect-data/"


class PretrainedModel:

    """ Generic class to download the pre-trained models

    Usage Example:

    chunker = ChunkerModel.get_instance()
    chunker2 = ChunkerModel.get_instance()
    print(chunker, chunker2)
    print("Local File path = ", chunker.get_file_path())
    files_models = chunker2.get_model_files()
    for idx, file_name in enumerate(files_models):
        print(str(idx) + ": " + file_name)

    """

    def __init__(self, model_name, sub_path, files):
        if isinstance(self, (BistModel, ChunkerModel, MrcModel, IntentModel, AbsaModel, NerModel)):
            if self._instance is not None:  # pylint: disable=no-member
                raise Exception("This class is a singleton!")
        self.model_name = model_name
        self.base_path = S3_PREFIX + sub_path
        self.files = files
        self.download_path = LIBRARY_OUT / "pretrained_models" / self.model_name
        self.model_files = []

    @classmethod
    # pylint: disable=no-member
    def get_instance(cls):
        """
        Static instance access method
        Args:
            cls (Class name): Calling class
        """
        if cls._instance is None:
            cls()  # pylint: disable=no-value-for-parameter
        return cls._instance

    def get_file_path(self):
        """
        Return local file path of downloaded model files
        """
        for filename in self.files:
            cached_file_path, need_downloading = cached_path(
                self.base_path + filename, self.download_path
            )
            if filename.endswith("zip"):
                if need_downloading:
                    print("Unzipping...")
                    uncompress_file(cached_file_path, outpath=self.download_path)
                    print("Done.")
        return self.download_path

    def get_model_files(self):
        """
        Return individual file names of downloaded models
        """
        for fileName in self.files:
            cached_file_path, need_downloading = cached_path(
                self.base_path + fileName, self.download_path
            )
            if fileName.endswith("zip"):
                if need_downloading:
                    print("Unzipping...")
                    uncompress_file(cached_file_path, outpath=self.download_path)
                    print("Done.")
                self.model_files.extend(zipfile_list(cached_file_path))
            else:
                self.model_files.extend([fileName])
        return self.model_files


# Model-specific classes developers instantiate where model has to be used


class BistModel(PretrainedModel):
    """
    Download and process (unzip) pre-trained BIST model
    """

    _instance = None
    sub_path = "models/dep_parse/"
    files = ["bist-pretrained.zip"]

    def __init__(self):
        super().__init__("bist", self.sub_path, self.files)
        BistModel._instance = self


class IntentModel(PretrainedModel):
    """
    Download and process (unzip) pre-trained Intent model
    """

    _instance = None
    sub_path = "models/intent/"
    files = ["model_info.dat", "model.h5"]

    def __init__(self):
        super().__init__("intent", self.sub_path, self.files)
        IntentModel._instance = self


class MrcModel(PretrainedModel):
    """
    Download and process (unzip) pre-trained MRC model
    """

    _instance = None
    sub_path = "models/mrc/"
    files = ["mrc_data.zip", "mrc_model.zip"]

    def __init__(self):
        super().__init__("mrc", self.sub_path, self.files)
        MrcModel._instance = self


class NerModel(PretrainedModel):
    """
    Download and process (unzip) pre-trained NER model
    """

    _instance = None
    sub_path = "models/ner/"
    files = ["model_v4.h5", "model_info_v4.dat"]

    def __init__(self):
        super().__init__("ner", self.sub_path, self.files)
        NerModel._instance = self


class AbsaModel(PretrainedModel):
    """
    Download and process (unzip) pre-trained ABSA model
    """

    _instance = None
    sub_path = "models/absa/"
    files = ["rerank_model.h5"]

    def __init__(self):
        super().__init__("absa", self.sub_path, self.files)
        AbsaModel._instance = self


class ChunkerModel(PretrainedModel):
    """
    Download and process (unzip) pre-trained Chunker model
    """

    _instance = None
    sub_path = "models/chunker/"
    files = ["model.h5", "model_info.dat.params"]

    def __init__(self):
        super().__init__("chunker", self.sub_path, self.files)
        ChunkerModel._instance = self
