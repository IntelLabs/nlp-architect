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

from setuptools import find_packages, setup

root = os.path.abspath(os.path.dirname(__file__))

# required packages for NLP Architect
with open("requirements.txt") as fp:
    reqs = []
    for r in fp.readlines():
        if "#" in r:
            continue
        reqs.append(r)
    install_requirements = reqs

everything = [
    "tensorflow_hub",
    "elasticsearch",
    "wordfreq",
    "newspaper3k",
    "pywikibot",
    "num2words",
    "bokeh",
    "pandas",
    "hyperopt",
    "termcolor",
]

dev = [
    "sphinx==1.8.5",
    "sphinx_rtd_theme",
    "flake8-html",
    "black",
    "pep8",
    "flake8",
    "pytest",
    "pytest-cov",
    "pytest-mock",
    "pytest-xdist",
    "pylint",
]
extras = {"all": everything, "dev": dev}

# read official README.md
with open("README.md", encoding="utf8") as fp:
    long_desc = fp.read()

with io.open(os.path.join(root, "nlp_architect", "version.py"), encoding="utf8") as f:
    version_f = {}
    exec(f.read(), version_f)
    version = version_f["NLP_ARCHITECT_VERSION"]

setup(
    name="nlp-architect",
    version=version,
    description="Intel AI Lab NLP and NLU research model library",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    keywords="NLP NLU deep learning natural language processing tensorflow pytorch",
    author="Intel AI Lab",
    author_email="nlp_architect@intel.com",
    url="https://github.com/NervanaSystems/nlp-architect",
    license="Apache 2.0",
    python_requires=">=3.6.*",
    packages=find_packages(
        exclude=["tests.*", "tests", "server.*", "server", "examples.*", "examples", "solutions.*", "solutions"]
    ),
    install_requires=install_requirements,
    extras_require=extras,
    scripts=["nlp_architect/nlp-train", "nlp_architect/nlp-inference"],
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: " + "Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: " + "Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
)
