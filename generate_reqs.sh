#!/bin/bash -e
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

# optimized pacakge links
tf_ver='1.6.0'
tf_to_install='tensorflow=='${tf_ver}
tf35='https://anaconda.org/intel/tensorflow/1.6.0/download/tensorflow-1.6.0-cp35-cp35m-linux_x86_64.whl'
tf36='https://anaconda.org/intel/tensorflow/1.6.0/download/tensorflow-1.6.0-cp36-cp36m-linux_x86_64.whl'


# detect python version and OS
pyver=`python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'`
platform='other'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
fi

ask_which_tf() {
  read -p 'Install Tensorflow MKL-DNN ? [Y/n] ' install_opt_tf
  retval=0
  case "${install_opt_tf}" in
    y|Y ) retval=1;;
    n|N ) retval=0;;
    "" ) retval=0;;
  esac
  if [ ${retval} == 1 ];
  then
    if [ ${pyver} == 3.5 ];
    then
      echo python 3.5 detected;
      tf_to_install=${tf35};
    elif [ ${pyver} == 3.6 ];
    then
      echo python 3.6 detected;
      tf_to_install=${tf36};
    fi
  fi
}

if [ ${pyver} == 3.5 ] || [ ${pyver} == 3.6 ] ;
then
  if [ ${platform} == 'linux' ] ;
  then
    ask_which_tf
  fi
fi

# read file
while read line
do
    line=${line/tensorflow*/$tf_to_install}
    echo "$line"
done < requirements.txt > _generated_reqs.txt
