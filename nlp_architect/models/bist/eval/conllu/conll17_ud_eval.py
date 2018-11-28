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
import sys

# Things that were changed from the original:
# - Added legal header
# - Reformatted code and variable names to conform with PEP8
# - Added pointer to 'weights.clas' file
# - Added run_conllu_eval()
# - Removed tests and command-line usage option
# - Removed unnecessary imports
# - Add pylint check disable flags

# !/usr/bin/env python
# CoNLL 2017 UD Parsing evaluation script.
#
# Compatible with Python 2.7 and 3.2+, can be used either as a module
# or a standalone executable.
#
# Copyright 2017 Institute of Formal and Applied Linguistics (UFAL),
# Faculty of Mathematics and Physics, Charles University, Czech Republic.
#
# Changelog:
# - [02 Jan 2017] Version 0.9: Initial release
# - [25 Jan 2017] Version 0.9.1: Fix bug in LCS alignment computation
# - [10 Mar 2017] Version 1.0: Add documentation and test
#                              Compare HEADs correctly using aligned words
#                              Allow evaluation with errorneous spaces in forms
#                              Compare forms in LCS case insensitively
#                              Detect cycles and multiple root nodes
#                              Compute AlignedAccuracy
# API usage
# ---------
# - load_conllu(file)
#   - loads CoNLL-U file from given file object to an internal representation
#   - the file object should return str on both Python 2 and Python 3
#   - raises UDError exception if the given file cannot be loaded
# - evaluate(gold_ud, system_ud)
#   - evaluate the given gold and system CoNLL-U files (loaded with
#     load_conllu)
#   - raises UDError if the concatenated tokens of gold and system file do not
#     match
#   - returns a dictionary with the metrics described above, each metrics
#     having three fields: precision, recall and f1
#
# Description of token matching
# -----------------------------
# In order to match tokens of gold file and system file, we consider the text
# resulting from concatenation of gold tokens and text resulting from
# concatenation of system tokens. These texts should match -- if they do not,
# the evaluation fails.
#
# If the texts do match, every token is represented as a range in this original
# text, and tokens are equal only if their range is the same.
#
# Description of word matching
# ----------------------------
# When matching words of gold file and system file, we first match the tokens.
# The words which are also tokens are matched as tokens, but words in
# multi-word tokens have to be handled differently.
#
# To handle multi-word tokens, we start by finding "multi-word spans".
# Multi-word span is a span in the original text such that
# - it contains at least one multi-word token
# - all multi-word tokens in the span (considering both gold and system ones)
#   are completely inside the span (i.e., they do not "stick out")
# - the multi-word span is as small as possible
#
# For every multi-word span, we align the gold and system words completely
# inside this span using LCS on their FORMs. The words not intersecting
# (even partially) any multi-word span are then aligned as tokens.
# pylint: disable=too-many-statements

# CoNLL-U column names
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = list(range(10))

WEIGHTS = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights.clas')


# UD Error is used when raising exceptions in this module
class UDError(Exception):
    pass


# Load given CoNLL-U file into internal representation
def load_conllu(file):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # Internal representation classes
    class UDRepresentation:
        # pylint: disable=too-few-public-methods
        def __init__(self):
            # Characters of all the tokens in the whole file.
            # Whitespace between tokens is not included.
            self.characters = []
            # List of UDSpan instances with start&end indices into `characters`
            self.tokens = []
            # List of UDWord instances.
            self.words = []
            # List of UDSpan instances with start&end indices into `characters`
            self.sentences = []

    class UDSpan:
        # pylint: disable=too-few-public-methods
        def __init__(self, start, end):
            self.start = start
            # Note that self.end marks the first position **after the end** of
            # span, so we can use characters[start:end] or range(start, end).
            self.end = end

    class UDWord:
        # pylint: disable=too-few-public-methods
        def __init__(self, span, columns, is_multiword):
            # Span of this word (or MWT, see below) within
            # ud_representation.characters.
            self.span = span
            # 10 columns of the CoNLL-U file: ID, FORM, LEMMA,...
            self.columns = columns
            # is_multiword==True means that this word is part of a multi-word
            # token.
            # In that case, self.span marks the span of the whole multi-word
            # token.
            self.is_multiword = is_multiword
            # Reference to the UDWord instance representing the HEAD (or None
            # if root).
            self.parent = None
            # Let's ignore language-specific deprel subtypes.
            self.columns[DEPREL] = columns[DEPREL].split(':')[0]

    ud = UDRepresentation()

    # Load the CoNLL-U file
    index, sentence_start = 0, None
    while True:
        line = file.readline()
        if not line:
            break
        line = line.rstrip("\r\n")

        # Handle sentence start boundaries
        if sentence_start is None:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            ud.sentences.append(UDSpan(index, 0))
            sentence_start = len(ud.words)
        if not line:
            # Add parent UDWord links and check there are no cycles
            def process_word(word):
                if word.parent == "remapping":
                    raise UDError("There is a cycle in a sentence")
                if word.parent is None:
                    head = int(word.columns[HEAD])
                    if head > len(ud.words) - sentence_start:
                        raise UDError(
                            "HEAD '{}' points outside of the sentence".format(
                                word.columns[HEAD]))
                    if head:
                        parent = ud.words[sentence_start + head - 1]
                        word.parent = "remapping"
                        process_word(parent)
                        word.parent = parent

            for word in ud.words[sentence_start:]:
                process_word(word)

            # Check there is a single root node
            if len([word for word in ud.words[sentence_start:] if
                    word.parent is None]) != 1:
                raise UDError("There are multiple roots in a sentence")

            # End the sentence
            ud.sentences[-1].end = index
            sentence_start = None
            continue

        # Read next token/word
        columns = line.split("\t")
        if len(columns) != 10:
            raise UDError(
                "The CoNLL-U line does not contain 10 tab-separated columns: "
                "'{}'".format(line))

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Delete spaces from FORM  so gold.characters == system.characters
        # even if one of them tokenizes the space.
        columns[FORM] = columns[FORM].replace(" ", "")
        if not columns[FORM]:
            raise UDError("There is an empty FORM in the CoNLL-U file")

        # Save token
        ud.characters.extend(columns[FORM])
        ud.tokens.append(UDSpan(index, index + len(columns[FORM])))
        index += len(columns[FORM])

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            try:
                start, end = list(map(int, columns[ID].split("-")))
            except Exception:
                raise UDError("Cannot parse multi-word token ID '{}'".format(
                    columns[ID]))

            for _ in range(start, end + 1):
                word_line = file.readline().rstrip("\r\n")
                word_columns = word_line.split("\t")
                if len(word_columns) != 10:
                    raise UDError(
                        "The CoNLL-U line does not contain 10 tab-separated "
                        "columns: '{}'".format(word_line))
                ud.words.append(
                    UDWord(ud.tokens[-1], word_columns, is_multiword=True))
        # Basic tokens/words
        else:
            try:
                word_id = int(columns[ID])
            except Exception:
                raise UDError("Cannot parse word ID '{}'".format(columns[ID]))
            if word_id != len(ud.words) - sentence_start + 1:
                raise UDError("Incorrect word ID '{}' for word '{}', expected"
                              " '{}'".format(columns[ID], columns[FORM],
                                             len(ud.words)
                                             - sentence_start + 1))

            try:
                head_id = int(columns[HEAD])
            except Exception:
                raise UDError("Cannot parse HEAD '{}'".format(columns[HEAD]))
            if head_id < 0:
                raise UDError("HEAD cannot be negative")

            ud.words.append(UDWord(ud.tokens[-1], columns, is_multiword=False))

    if sentence_start is not None:
        raise UDError("The CoNLL-U file does not end with empty line")

    return ud


# Evaluate the gold and system treebanks (loaded using load_conllu).
def evaluate(gold_ud, system_ud, deprel_weights=None):
    # pylint: disable=too-many-locals
    class Score:
        # pylint: disable=too-few-public-methods
        def __init__(self, gold_total, system_total, correct,
                     aligned_total=None):
            self.precision = correct / system_total if system_total else 0.0
            self.recall = correct / gold_total if gold_total else 0.0
            self.f1 = 2 * correct / (system_total + gold_total) \
                if system_total + gold_total else 0.0
            self.aligned_accuracy = \
                correct / aligned_total if aligned_total else aligned_total

    class AlignmentWord:
        # pylint: disable=too-few-public-methods
        def __init__(self, gold_word, system_word):
            self.gold_word = gold_word
            self.system_word = system_word
            self.gold_parent = None
            self.system_parent_gold_aligned = None

    class Alignment:
        def __init__(self, gold_words, system_words):
            self.gold_words = gold_words
            self.system_words = system_words
            self.matched_words = []
            self.matched_words_map = {}

        def append_aligned_words(self, gold_word, system_word):
            self.matched_words.append(AlignmentWord(gold_word, system_word))
            self.matched_words_map[system_word] = gold_word

        def fill_parents(self):
            # We represent root parents in both gold and system data by '0'.
            # For gold data, we represent non-root parent by corresponding gold
            # word.
            # For system data, we represent non-root parent by either gold word
            # aligned
            # to parent system nodes, or by None if no gold words is aligned to
            # the parent.
            for words in self.matched_words:
                words.gold_parent = words.gold_word.parent if \
                    words.gold_word.parent is not None else 0
                words.system_parent_gold_aligned = self.matched_words_map.get(
                    words.system_word.parent, None) \
                    if words.system_word.parent is not None else 0

    def lower(text):
        if sys.version_info < (3, 0) and isinstance(text, str):
            return text.decode("utf-8").lower()
        return text.lower()

    def spans_score(gold_spans, system_spans):
        correct, gi, si = 0, 0, 0
        while gi < len(gold_spans) and si < len(system_spans):
            if system_spans[si].start < gold_spans[gi].start:
                si += 1
            elif gold_spans[gi].start < system_spans[si].start:
                gi += 1
            else:
                correct += gold_spans[gi].end == system_spans[si].end
                si += 1
                gi += 1

        return Score(len(gold_spans), len(system_spans), correct)

    def alignment_score(alignment, key_fn, weight_fn=lambda w: 1):
        gold, system, aligned, correct = 0, 0, 0, 0

        for word in alignment.gold_words:
            gold += weight_fn(word)

        for word in alignment.system_words:
            system += weight_fn(word)

        for words in alignment.matched_words:
            aligned += weight_fn(words.gold_word)

        if key_fn is None:
            # Return score for whole aligned words
            return Score(gold, system, aligned)

        for words in alignment.matched_words:
            if key_fn(words.gold_word, words.gold_parent) == key_fn(
                    words.system_word,
                    words.system_parent_gold_aligned):
                correct += weight_fn(words.gold_word)

        return Score(gold, system, correct, aligned)

    def beyond_end(words, i, multiword_span_end):
        if i >= len(words):
            return True
        if words[i].is_multiword:
            return words[i].span.start >= multiword_span_end
        return words[i].span.end > multiword_span_end

    def extend_end(word, multiword_span_end):
        if word.is_multiword and word.span.end > multiword_span_end:
            return word.span.end
        return multiword_span_end

    def find_multiword_span(gold_words, system_words, gi, si):
        # We know gold_words[gi].is_multiword or system_words[si].is_multiword.
        # Find the start of the multiword span (gs, ss), so the multiword span
        # is minimal.
        # Initialize multiword_span_end characters index.
        if gold_words[gi].is_multiword:
            multiword_span_end = gold_words[gi].span.end
            if not system_words[si].is_multiword and system_words[si].span.start < \
                    gold_words[gi].span.start:
                si += 1
        else:  # if system_words[si].is_multiword
            multiword_span_end = system_words[si].span.end
            if not gold_words[gi].is_multiword and gold_words[gi].span.start < \
                    system_words[si].span.start:
                gi += 1
        gs, ss = gi, si

        # Find the end of the multiword span
        # (so both gi and si are pointing to the word following the multiword
        # span end).
        while not beyond_end(gold_words, gi, multiword_span_end) or \
                not beyond_end(system_words, si, multiword_span_end):
            gold_start = gold_words[gi].span.start
            sys_start = system_words[si].span.start
            if gi < len(gold_words) and (si >= len(system_words) or gold_start <= sys_start):
                multiword_span_end = extend_end(gold_words[gi], multiword_span_end)
                gi += 1
            else:
                multiword_span_end = extend_end(system_words[si], multiword_span_end)
                si += 1
        return gs, ss, gi, si

    def compute_lcs(gold_words, system_words, gi, si, gs, ss):
        # pylint: disable=too-many-arguments
        lcs = [[0] * (si - ss) for _ in range(gi - gs)]
        for g in reversed(list(range(gi - gs))):
            for s in reversed(list(range(si - ss))):
                if lower(gold_words[gs + g].columns[FORM]) == lower(
                        system_words[ss + s].columns[FORM]):
                    lcs[g][s] = 1 + (lcs[g + 1][s + 1] if
                                     g + 1 < gi - gs and s + 1 < si - ss
                                     else 0)
                lcs[g][s] = max(lcs[g][s],
                                lcs[g + 1][s] if g + 1 < gi - gs else 0)
                lcs[g][s] = max(lcs[g][s],
                                lcs[g][s + 1] if s + 1 < si - ss else 0)
        return lcs

    def align_words(gold_words, system_words):
        alignment = Alignment(gold_words, system_words)

        gi, si = 0, 0
        while gi < len(gold_words) and si < len(system_words):
            if gold_words[gi].is_multiword or system_words[si].is_multiword:
                # A: Multi-word tokens => align via LCS within the whole
                # "multiword span".
                gs, ss, gi, si = find_multiword_span(gold_words, system_words,
                                                     gi, si)

                if si > ss and gi > gs:
                    lcs = compute_lcs(gold_words, system_words, gi, si, gs, ss)

                    # Store aligned words
                    s, g = 0, 0
                    while g < gi - gs and s < si - ss:
                        if lower(gold_words[gs + g].columns[FORM]) == lower(
                                system_words[ss + s].columns[FORM]):
                            alignment.append_aligned_words(gold_words[gs + g],
                                                           system_words[
                                                               ss + s])
                            g += 1
                            s += 1
                        elif lcs[g][s] == (
                                lcs[g + 1][s] if g + 1 < gi - gs else 0):
                            g += 1
                        else:
                            s += 1
            else:
                # B: No multi-word token => align according to spans.
                if (gold_words[gi].span.start, gold_words[gi].span.end) == (
                        system_words[si].span.start,
                        system_words[si].span.end):
                    alignment.append_aligned_words(gold_words[gi],
                                                   system_words[si])
                    gi += 1
                    si += 1
                elif gold_words[gi].span.start <= system_words[si].span.start:
                    gi += 1
                else:
                    si += 1

        alignment.fill_parents()

        return alignment

    # Check that underlying character sequences do match
    if gold_ud.characters != system_ud.characters:
        index = 0
        while gold_ud.characters[index] == system_ud.characters[index]:
            index += 1

        raise UDError(
            "The concatenation of tokens in gold file and in system file "
            "differ!\n"
            + "First 20 differing characters in gold file: '{}' and system file:"
            " '{}'".format(
                "".join(gold_ud.characters[index:index + 20]),
                "".join(system_ud.characters[index:index + 20])
            )
        )

    # Align words
    alignment = align_words(gold_ud.words, system_ud.words)

    # Compute the F1-scores
    result = {
        "Tokens": spans_score(gold_ud.tokens, system_ud.tokens),
        "Sentences": spans_score(gold_ud.sentences, system_ud.sentences),
        "Words": alignment_score(alignment, None),
        "UPOS": alignment_score(alignment, lambda w, parent: w.columns[UPOS]),
        "XPOS": alignment_score(alignment, lambda w, parent: w.columns[XPOS]),
        "Feats": alignment_score(alignment,
                                 lambda w, parent: w.columns[FEATS]),
        "AllTags": alignment_score(alignment, lambda w, parent: (
            w.columns[UPOS], w.columns[XPOS], w.columns[FEATS])),
        "Lemmas": alignment_score(alignment,
                                  lambda w, parent: w.columns[LEMMA]),
        "UAS": alignment_score(alignment, lambda w, parent: parent),
        "LAS": alignment_score(alignment,
                               lambda w, parent: (parent, w.columns[DEPREL])),
    }

    # Add WeightedLAS if weights are given
    if deprel_weights is not None:
        def weighted_las(word):
            return deprel_weights.get(word.columns[DEPREL], 1.0)

        result["WeightedLAS"] = alignment_score(alignment, lambda w, parent: (
            parent, w.columns[DEPREL]), weighted_las)

    return result


def load_deprel_weights(weights_file):
    if weights_file is None:
        return None

    deprel_weights = {}
    with open(weights_file) as f:
        for line in f:
            # Ignore comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            columns = line.rstrip("\r\n").split()
            if len(columns) != 2:
                raise ValueError(
                    "Expected two columns in the UD Relations weights file on line"
                    " '{}'".format(
                        line))

            deprel_weights[columns[0]] = float(columns[1])

    return deprel_weights


def load_conllu_file(path):
    with open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {})) \
            as _file:
        return load_conllu(_file)


def evaluate_wrapper(gold_file: str, system_file: str, weights_file: str):
    # Load CoNLL-U files
    gold_ud = load_conllu_file(gold_file)
    system_ud = load_conllu_file(system_file)

    # Load weights if requested
    deprel_weights = load_deprel_weights(weights_file)

    return evaluate(gold_ud, system_ud, deprel_weights)


def run_conllu_eval(gold_file, test_file, weights_file=WEIGHTS, verbose=True):
    # Use verbose if weights are supplied
    if weights_file is not None and not verbose:
        verbose = True

    # Evaluate
    evaluation = evaluate_wrapper(gold_file, test_file, weights_file)

    # Write the evaluation to file
    with open(test_file[:test_file.rindex('.')] + '_eval.txt', 'w') as out_file:
        if not verbose:
            out_file.write("LAS F1 Score: {:.2f}".format(100 * evaluation["LAS"].f1) + '\n')
        else:
            metrics = ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "Feats",
                       "AllTags", "Lemmas", "UAS", "LAS"]
            if weights_file is not None:
                metrics.append("WeightedLAS")

            out_file.write("Metrics    | Precision |    Recall |  F1 Score | AligndAcc" + '\n')
            out_file.write("-----------+-----------+-----------+-----------+-----------" + '\n')
            for metric in metrics:
                out_file.write("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                    metric,
                    100 * evaluation[metric].precision,
                    100 * evaluation[metric].recall,
                    100 * evaluation[metric].f1,
                    "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy)
                    if evaluation[metric].aligned_accuracy is not None else ""
                ) + '\n')
