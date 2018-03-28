Installation
############

Requirements
============

The Intel® Nervana™ NLP AI-Lab requires **Python 3.5+** running on a
Linux* or UNIX-based OS. Each model contains its own installation requirements, but in general,
before installing, ensure your system has recent updates of the following packages:

.. csv-table::
   :header: "Ubuntu* 16.04+ or CentOS* 7.4+", "Mac OS X*", "Description"
   :widths: 20, 20, 42
   :escape: ~

   python-pip, pip, Tool to install Python dependencies
   python-virtualenv (*), virtualenv (*), Allows creation of isolated environments ((*): This is required only for Python 2.7 installs. With Python3: test for presence of ``venv`` with ``python3 -m venv -h``)
   libhdf5-dev, h5py, Enables loading of hdf5 formats
   libyaml-dev, pyaml, Parses YAML format inputs
   pkg-config, pkg-config, Retrieves information about installed libraries
