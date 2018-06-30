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
# pylint: disable=invalid-name
import numpy as np


# Things that were changed from the original:
# - Reformatted code and variable names to conform with PEP8
# - Added legal header


# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/


def parse_proj(scores, gold=None):
    # pylint: disable=too-many-locals
    """
    Parse using Eisner's algorithm.
    """
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1  # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    incomplete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    complete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)  # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)  # s, t, direction (right=1).

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in range(1, N + 1):
        for s in range(N - k + 1):
            t = s + k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s + 1):(t + 1), t, 0] + scores[
                t, s] + (0.0 if gold is not None and gold[s] == t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s + 1):(t + 1), t, 0] + scores[
                s, t] + (0.0 if gold is not None and gold[t] == s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s + 1):(t + 1), 1] + complete[(s + 1):(t + 1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    # value = complete[0][N][1]
    heads = [-1 for _ in range(N + 1)]  # -np.ones(N+1, dtype=int)
    _backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)
    value_proj = 0.0
    for m in range(1, N + 1):
        h = heads[m]
        value_proj += scores[h, m]
    return heads


# pylint: disable=too-many-arguments
def _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    """
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start
     position, an end position, and a direction flag (0 means left, 1 means
     right). This array contains the arg-maxes of each step in the Eisner
     algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start
     position, an end position, and a direction flag (0 means left, 1 means
     right). This array contains the arg-maxes of each step in the Eisner
     algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder
      for storing the head of each word.
    """
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            _backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
        _backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
        return

    r = incomplete_backtrack[s][t][direction]
    if direction == 0:
        heads[s] = t
        _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
        _backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
        return
    heads[t] = s
    _backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
    _backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
    return
