#!/usr/bin/env python
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
import os
import platform
import subprocess
import sys

from setuptools import setup, find_packages

from nlp_architect.version import nlp_architect_version

# required packages for NLP Architect
requirements = [
    "dynet==2.0.2",
    "spacy<2.0.12",
    "nltk",
    "gensim",
    "sklearn",
    "scipy",
    "numpy<=1.14.5",
    "tensorflow_hub",
    "elasticsearch",
    "fastText@git+https://github.com/facebookresearch/fastText.git#egg=fastText",
    "newspaper3k",
    "wordfreq",
    "seqeval",
    "pywikibot",
    "num2words",
    "hyperopt",
    "pandas",
    "tqdm",
    "ftfy",
    "bokeh",
    "six",
    "future",
    "requests",
    "termcolor",
    "pillow",
    "setuptools==39.1.0",
    "hug",
    "falcon",
    "falcon_multipart",
    "sphinx",
    "sphinx_rtd_theme",
]

# required packages for testing
test_requirements = [
    'pep8',
    'flake8',
    'pytest',
    'pytest-cov',
    'pytest-mock',
    'pylint',
]

# check if GPU available
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
out = p.communicate()[0].decode('UTF-8')
gpu_available = len(out) > 0

# check python version
py3_ver = int(platform.python_version().split('.')[1])

# Tensorflow version (make sure CPU/MKL/GPU versions exist before changing)
tf_version = '1.10.0'
tf_mkl_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-{}-cp3{}-cp3{}m-linux_x86_64.whl'

# default TF is CPU
chosen_tf = 'tensorflow=={}'.format(tf_version)
# check system is linux for MKL/GPU backends
if 'linux' in sys.platform:
    system_type = 'linux'
    tf_be = os.getenv('NLP_ARCHITECT_BE', False)
    if tf_be and 'mkl' == tf_be.lower():
        if py3_ver == 5 or py3_ver == 6:
            chosen_tf = tf_mkl_url.format(tf_version, py3_ver, py3_ver)
    elif tf_be and 'gpu' == tf_be.lower() and gpu_available:
        chosen_tf = 'tensorflow-gpu=={}'.format(tf_version)
requirements.append(chosen_tf)

with open('README.md', encoding='UTF-8') as fp:
    long_desc = fp.read()

setup(name='nlp_architect',
      version=nlp_architect_version(),
      description='NLP Architect by Intel AI Lab: Python library for exploring the '
                  'state-of-the-art deep learning topologies and techniques for natural language '
                  'processing and natural language understanding',
      long_description=long_desc,
      keywords='NLP NLU deep learning natural language processing tensorflow keras dynet',
      author='Intel AI Lab',
      author_email='nlp_architect@intel.com',
      url='https://github.com/NervanaSystems/nlp-architect',
      license='Apache 2.0',
      python_requires='>=3.5.*',
      packages=find_packages(exclude=['tests.*', 'tests', '*.tests', '*.tests.*',
                                      'examples.*', 'examples', '*.examples', '*.examples.*']),
      install_requires=requirements + test_requirements,
      scripts=['nlp_architect/nlp_architect'],
      include_package_data=True,
      package_data={
          'server': ['services.json'],
          'nlp_architect.utils.resources': ['preposition_en.json', 'pronoun_en.json',
                                            'stop_words_en.json', 'stopwords.txt']
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: POSIX',
          'Operating System :: MacOS :: MacOS X',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: ' +
          'Artificial Intelligence',
          'Topic :: Scientific/Engineering :: ' +
          'Natural Language Processing',
          'Topic :: Scientific/Engineering :: ' +
          'Natural Language Understanding',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: System :: Distributed Computing']
      )
