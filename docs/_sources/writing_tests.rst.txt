.. ---------------------------------------------------------------------------
.. Copyright 2017-2018 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Tests Conventions
=================

To ensure code stability, developers are required to write tests for contributed modules,
according to these guidelines:

* Tests are written using `pytest`_

* All test functions must have docstrings

* Tests modules should be placed in the ./tests directory

.. _pytest: https://docs.pytest.org/en/latest/


Test Types
----------

* **Regression tests** – check that claimed model/pipeline end-to-end functionality is preserved after changes. Test sub-components where appropriate.

* **Util tests** – test functionality of utility modules.

* **Framework tests** – test http service input/output


Running Tests
-------------

Before creating a pull request, ensure that your test pass.
To a test module, issue the following command:

.. code:: bash

    py.test tests/test_module.py
