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
from setuptools import setup, find_packages

# Define version information
VERSION = '0.2'
FULLVERSION = VERSION

requirements = [
]

setup(name='nlp_architect',
      version=VERSION,
      description="Intel AI Lab NLP Deep Learning framework for Research",
      author='Intel AI Lab NLP',
      author_email='nlp_architect@intel.com',
      url='http://ai.intel.com',
      license='License :: OSI Approved :: Apache Software License',
      packages=find_packages(),
      install_requires=requirements,
      include_package_data=True,
      classifiers=['Development Status :: 3 - Alpha',
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
                   'Topic :: System :: Distributed Computing'])
