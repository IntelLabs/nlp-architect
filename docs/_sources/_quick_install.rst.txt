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

Quick Install
-------------
Select the desired configuration of your system:

.. raw:: html

    <div id="app" class="wy-table-responsive">
    <table class="docutils option-list installation_table" frame="void" rules="none">
        <colgroup><col class="option">
        <col class="description">
        </colgroup><tbody valign="top">
        <tr><td class="option-group">
        <kbd><span class="option">
            <strong>Install from</strong>
        </span></kbd></td>
        <td>
            <label class="radio">
              <input v-model="form.source" type="radio" value="1">Pip
            </label>
            <label class="radio">
              <input v-model="form.source" type="radio" value="0" checked>GitHub
            </label>
        </td></tr>
        <tr><td class="option-group">
        <kbd><span class="option">
            <strong>Create virtualenv?</strong>
        </span></kbd></td>
        <td>
            <label class="radio">
              <input v-model="form.with_env" type="radio" value="1">Yes
            </label>
            <label class="radio">
              <input v-model="form.with_env" type="radio" value="0" checked>No
            </label>
        </td></tr>
        <tr><td class="option-group">
        <kbd><span class="option">
            <strong>Backend</strong>
        </span></kbd></td>
        <td>
            <label class="radio">
              <input v-model="form.backend" type="radio" value="CPU">CPU
            </label>
            <label class="radio">
              <input v-model="form.backend" type="radio" value="MKL" checked>MKL
            </label>
            <label class="radio">
              <input v-model="form.backend" type="radio" value="GPU" checked>GPU
            </label>
        </td></tr>
        <tr><td class="option-group">
        <kbd><span class="option">
            <strong>Install in developer mode?</strong>
        </span></kbd></td>
        <td>
        <label class="radio">
              <input v-model="form.inst_type" type="radio" value="0">Yes
            </label>
            <label class="radio">
              <input v-model="form.inst_type" type="radio" value="1" checked>No
            </label>
        </td></tr>
        </tbody>
    </table>

Run the following commands to install NLP Architect:

.. raw:: html

    <div class="code python highlight-default notranslate"><div class="highlight">
    <pre v-html="get_commands()">
    </pre></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/vue"></script>
    <script src="_static/install.js"></script>

It is recommended to install NLP Architect in development mode to utilize all its features, examples and solutions.