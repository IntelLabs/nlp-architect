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

Contribution Process
====================

1. File an issue (to track your contribution):

   * Create an issue on GitHub:
     https://github.com/NervanaSystems/nlp-architect/issues

2. Clone and/or update your checked out copy of nlp-architect to ensure you have the
   most recent commits from the master branch:

  .. code-block:: bash

      git clone https://github.com/NervanaSystems/nlp-architect.git
      cd nlp-architect
      git fetch origin
      git checkout master
      git pull

3. Create a new feature branch for your work and switch to it.  Give it a
   meaningful name related to the task(s) at hand:

  .. code-block:: bash

      git checkout -b my_new_feature_branch

4. Ideally you'd start by creating one or more unit tests with the
   functionality you expect your new feature to perform.  These should reside
   under the appropriate tests subdirectory of whatever you are changing.
   Then hack away at the code until you feel your feature is complete.  Once
   satisfied, run the code through the following checks:

  .. code-block:: bash

      nlp_architect test   # ensure all are OK
      nlp_architect style  # ensure there are no style related issues

5. If necessary you may want to update and/or rebuild the documentation.
   This all exists under doc/source and is in
   `Sphinx reStructuredText format <http://sphinx-doc.org/rest.html>`_:

  .. code-block:: bash

      nlp_architect doc   # builds the doc and starts a local server directly

6. Commit your changes and push your feature branch to your GitHub fork.  Be
   sure to add a descriptive message and reference the GitHub issue associated
   with your task (ex. #1).  You will also want to rebase your commits down to
   a single sensible commit to make things clean for the merge process:

  .. code-block:: bash

      git add my_updated_files
      git commit -m "Added new awesome functionality.  Closes issue #1"
      git push origin my_new_feature_branch

7. Create a new pull request to get your feature branch merged into master for
   others to use.  You'll first need to ensure your feature branch contains the
   latest changes from master.  Furthermore, internal devs will need to assign
   the request to someone else for a code review.  You must also ensure there
   are no errors when run through the items defined in step 4.

  .. code-block:: bash

      # (external contribs): make a new pull request:
      https://github.com/NervanaSystems/nlp-architect/pulls

      # merge latest master changes into your feature branch
      git fetch origin
      git checkout master
      git pull origin master
      git checkout my_new_feature_branch
      git merge master  # you may need to manually resolve any merge conflicts

8. If there are issues you can continue to push commits to your feature branch
   by following step 6.  They will automatically be added to this same merge
   request.

9. Once your change has been successfully merged, you can remove the source
   branch and ensure your local copy is up to date:

  .. code-block:: bash

      git fetch origin
      git checkout master
      git pull
      git branch -d my_new_feature_branch
      git branch -d -r origin/my_new_feature_branch

10. Give yourself a high five for a job well done!
