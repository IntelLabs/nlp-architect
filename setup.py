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
import io
import os
import subprocess
import sys

from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))


# required packages for NLP Architect
with open('requirements.txt') as fp:
    install_requirements = fp.readlines()

# check if GPU available
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
out = p.communicate()[0].decode('utf8')
gpu_available = len(out) > 0

# Tensorflow version (make sure CPU/MKL/GPU versions exist before changing)
for r in install_requirements:
    if r.startswith('tensorflow=='):
        tf_version = r.split('==')[1]

# default TF is CPU
chosen_tf = 'tensorflow=={}'.format(tf_version)
# check system is linux for MKL/GPU backends
if 'linux' in sys.platform:
    system_type = 'linux'
    tf_be = os.getenv('NLP_ARCHITECT_BE', False)
    if tf_be and 'mkl' == tf_be.lower():
        chosen_tf = 'intel-tensorflow=={}'.format(tf_version)
    elif tf_be and 'gpu' == tf_be.lower() and gpu_available:
        chosen_tf = 'tensorflow-gpu=={}'.format(tf_version)

for r in install_requirements:
    if r.startswith('tensorflow=='):
        install_requirements[install_requirements.index(r)] = chosen_tf

with open('README.md', encoding='utf8') as fp:
    long_desc = fp.read()

with io.open(os.path.join(root, 'nlp_architect', 'version.py'), encoding='utf8') as f:
    version_f = {}
    exec(f.read(), version_f)
    version = version_f['NLP_ARCHITECT_VERSION']

setup(name='nlp-architect',
      version=version,
      description='Intel AI Lab\'s open-source NLP and NLU research library',
      long_description=long_desc,
      long_description_content_type='text/markdown',
      keywords='NLP NLU deep learning natural language processing tensorflow keras dynet',
      author='Intel AI Lab',
      author_email='nlp_architect@intel.com',
      url='https://github.com/NervanaSystems/nlp-architect',
      license='Apache 2.0',
      python_requires='>=3.6.*',
      packages=find_packages(exclude=['tests.*', 'tests', '*.tests', '*.tests.*',
                                      'examples.*', 'examples', '*.examples', '*.examples.*']),
      install_requires=install_requirements,
      scripts=['nlp_architect/nlp_architect'],
      include_package_data=True,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: ' +
          'Artificial Intelligence',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: ' +
          'Python Modules',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Operating System :: POSIX',
          'Operating System :: MacOS :: MacOS X',
      ]
      )
