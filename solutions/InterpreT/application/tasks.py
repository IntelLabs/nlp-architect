# ******************************************************************************
# Copyright 2020-2021 Intel Corporation
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
""" Module for defining a "Task" (e.g. ABSA, WSC) for InterpreT.

The tasks defined in this module will be used to configure the
InterpreT app.
"""

from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np


class Task(ABC):
    """Abstract class for defining tasks.

    To create your own task, please refer to the provided ABSATask and WSCTask as examples for how to do so.
    There are 4 abstract methods that must be defined: _init_tsne_layout(), _init_table_cols_layout(),
    get_val_to_label_map(), get_tsne_rows(). These definitions will all depend on how the input dataframe
    is structured.

    Attributes:
        mode: 'normal' if a single model is loaded, else 'compare'
        num_layers: number of transformer layers in the model
        model_ids: ["model1"] or ["model1", "model2"], used internally
        model_names: a list of the given model names to be displayed in app (correspond to above model_ids)
        collateral_paths: list of filepaths to the model collaterals (attn, hidden_states)
        df_paths: list of filepaths to model dataframe (everything else)

        tsne_internal_options: list of column names from the dataframe to color the tsne plot with
        tsne_displayed_options: list of names to display corresponding to the above plot options
        tsne_plot_description: string title for the tsne plot
        attn_map_options: list of options to display for attention map plot
        head_matrix_options: list of options to display for head matrix plot
        table_cols: list of columns to display for the summary table

        map_model_id_to_name: dictionary mapping "model1"/"model2" (used internally) to map_model_to_path
        map_model_to_full_df:dict mapping model_ids to dataframes (after _preprocess_df)
        map_model_to_df: dict mapping model_ids to dataframes (after _preprocess_df and _process_all_df)
        map_model_to_collateral: dict mapping model_ids to collateral pahts
    """

    def __init__(
        self,
        mode: str,
        num_layers: int,
        model_names: List[str],
        collaterals_paths: List[str],
        df_paths: List[str],
    ):
        # General attributes and filepaths
        self.mode = mode
        self.num_layers = num_layers
        self.model_ids = [f"model{i + 1}" for i in range(len(collaterals_paths))]
        self.model_names = model_names if model_names is not None else self.model_ids
        self.collaterals_paths = collaterals_paths
        self.df_paths = df_paths

        # Mappings to and from user-provided model names (e.g. "BERT", "LiBERT") to internally used model name ("model1", "model2")
        self.map_model_id_to_name, self.map_model_name_to_id = self._init_model_maps()

        # Layout options
        (
            self.tsne_internal_options,
            self.tsne_displayed_options,
            self.tsne_plot_description,
        ) = self._init_tsne_layout()
        self.attn_map_options = self._init_attn_layout()
        self.head_matrix_options = self._init_head_matrix_layout()
        self.table_cols = self._init_table_cols_layout()

        # Mappings to from model names to different data sources (dataframe, collateral)
        self.map_model_to_full_df, self.map_model_to_df = self._init_df_maps()
        self.map_model_to_collateral = self._init_collateral_map()

    @abstractmethod
    def _init_tsne_layout(self) -> List[str]:
        """Get layout options for the tsne plot."""
        pass

    def _init_attn_layout(self) -> List[str]:
        """Get layout options for the attention visualization."""
        attn_map_options = []
        for model_name in self.model_names:
            attn_map_options += [
                model_name,
            ]
        if self.mode == "compare":
            attn_map_options += ["delta"]
        return attn_map_options

    def _init_head_matrix_layout(self) -> List[str]:
        """Get layout options for the head matrix summary plot."""
        head_matrix_options = []
        for model_name in self.model_names:
            head_matrix_options += [
                f"{model_name}_std",
            ]
        if self.mode == "compare":
            head_matrix_options += ["delta_std"]
        return self._add_head_matrix_options(head_matrix_options)

    def _add_head_matrix_options(self, head_matrix_options: List[str]) -> List[str]:
        """Add additional task-specific head matrix summary option labels (e.g. custom metrics)."""
        return head_matrix_options

    @abstractmethod
    def _init_table_cols_layout(self) -> List[str]:
        """Get column labels for the summary table."""
        pass

    def _init_model_maps(self) -> Dict[str, str]:
        """Creates mapping between internal model ids ('model1'/'model2') and given model names."""
        map_model_id_to_name = {
            model_id: model_name for model_id, model_name in zip(self.model_ids, self.model_names)
        }
        map_model_name_to_id = {
            model_name: model_id for model_id, model_name in zip(self.model_ids, self.model_names)
        }
        return map_model_id_to_name, map_model_name_to_id

    def _init_df_maps(self) -> Tuple[Dict[str, pd.DataFrame]]:
        """Reads and processes dataframe for use by application."""
        full_dfs = {}
        tsne_dfs = {}
        for model_id, df in zip(self.model_ids, self.df_paths):
            full_dfs[model_id] = pd.read_csv(df)
            tsne_dfs[model_id] = self._preprocess_df(
                full_dfs[model_id]
            )  # model-specific processing
        tsne_dfs = self._process_all_df(tsne_dfs)  # model-agnostic processing
        return full_dfs, tsne_dfs

    def _init_collateral_map(self) -> Dict[str, Dict[str, Union[np.array, List[str]]]]:
        """Returns dict mapping model names to collaterals (attentions and hidden states)."""
        collaterals = {}
        for model_id, collateral in zip(self.model_ids, self.collaterals_paths):
            collaterals[model_id] = torch.load(collateral)
        return collaterals

    def get_dropdown_to_df_col_map(self) -> Dict[str, str]:
        """Returns dict mapping values for the column to labels based on dropdown_color_option"""
        return {
            displayed_option: internal_option
            for displayed_option, internal_option in zip(
                self.tsne_displayed_options, self.tsne_internal_options
            )
        }

    @abstractmethod
    def get_val_to_label_map(self, dropdown_color_option: str) -> Dict[float, str]:
        """Returns dict mapping t-SNE plot color option to a dictionary of (val, label) k/v pairs, for categorical plotting."""
        pass

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies model-specific preprocessing to a dataframe."""
        return df

    def _process_all_df(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Processes list of dataframes to be used by application."""
        return dfs

    @abstractmethod
    def get_tsne_rows(
        self, saved_click: Dict[str, pd.DataFrame], model_selector_val: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get rows from the tsne dataframe based on click."""
        pass

    def get_attn_map(self, attn_map_option: str, example_id: int) -> Tuple[np.array, List[str]]:
        """Get attention matrix for a specific summary option and example indices."""
        # Collect attentions and tokens for each model
        for model_id, model_name in zip(self.model_ids, self.model_names):
            if attn_map_option == model_name:
                tokens = self.map_model_to_collateral[model_id][example_id]["tokens"]
                attns = self.map_model_to_collateral[model_id][example_id]["attentions"]

        # Compute attention delta if comparing two models
        if attn_map_option == "delta":
            tokens = self.map_model_to_collateral["model1"][example_id]["tokens"]
            attns1 = np.stack(self.map_model_to_collateral["model1"][example_id]["attentions"])
            attns2 = np.stack(self.map_model_to_collateral["model2"][example_id]["attentions"])
            attns = attns1 - attns2

        attns, tokens = self._unpad_tokens(attns, tokens)
        return attns, tokens

    def get_head_matrix(self, head_matrix_option: str, example_ids: List[int]) -> np.array:
        """Get summary plot matrix for a task-AGNOSTIC summary option and example indices."""
        z = None
        attentions_dict = {}

        # Compute std(attn) if given the corresponding option
        for model_id, model_name in zip(self.model_ids, self.model_names):
            if head_matrix_option == f"{model_name}_std":
                attns = [
                    np.stack(self.map_model_to_collateral[model_id][example_id]["attentions"])
                    for example_id in example_ids
                ]
                attentions_dict[model_id] = attns
                z = np.mean([attn.std(axis=(2, 3)) for attn in attns], axis=0)

        # Compute std(attn2 - attn1) if given the corresponding option
        if self.mode == "compare":
            if head_matrix_option == "delta_std":
                attns1 = attentions_dict["model1"]
                attns2 = attentions_dict["model2"]
                attns_delta = [attn2 - attn1 for (attn1, attn2) in zip(attns1, attns2)]
                z = np.mean([attn.std(axis=(2, 3)) for attn in attns_delta], axis=0)

        # If head_matrix_option not task-agnostic
        if z is None:
            z = self.get_specific_head_matrix(head_matrix_option, example_ids)

        assert z is not None, "z is None"

        return z

    def get_specific_head_matrix(self, head_matrix_option: str, example_ids: List[int]) -> np.array:
        """Get summary plot matrix for a task-SPECIFIC summary options and example indices (e.g. custom metrics)."""
        return None

    def get_name(self) -> str:
        """Returns name of task, as defined by subclass."""
        return self.name

    def get_df_col_to_plot(self) -> str:
        """Returns the DataFrame column to be used for the t-SNE plot, as defined by subclass."""
        return self.df_col_to_plot

    @abstractmethod
    def get_summary_table(self, saved_click: pd.DataFrame) -> Dict[str, Union[str, int]]:
        """Updates summary table with values associated with the saved click."""
        pass

    def _unpad_tokens(self, attns: np.array, tokens: List[str]) -> Tuple[np.array, List[str]]:
        """Unpad tokens if we need to."""
        return attns, tokens


class WSCTask(Task):
    def __init__(self, *args, **kwargs):
        super(WSCTask, self).__init__(*args, **kwargs)
        self.name = "wsc"
        self.df_col_to_plot = "span_token"

    def _init_tsne_layout(self) -> List[str]:
        """Get layout options for the tsne plot."""
        tsne_internal_options = ["acc", "tp", "pred", "target"]
        tsne_displayed_options = ["acc_coloring", "tp_coloring", "pred_coloring", "target_coloring"]
        tsne_plot_description = "Plotting Coreferent Span Tokens"
        return tsne_internal_options, tsne_displayed_options, tsne_plot_description

    def _add_head_matrix_options(self, head_matrix_options: List[str]) -> List[str]:
        """Add coreference intensity metric for WSC."""
        for model_name in self.model_names:
            head_matrix_options += [
                f"{model_name}_coreference_intensity",
                f"{model_name}_accuracy_based_on_head",
            ]
        if self.mode == "compare":
            head_matrix_options += ["coreference_intensity_delta"]
        return head_matrix_options

    def _init_table_cols_layout(self) -> List[str]:
        """Get column labels for the summary table for WSC."""
        table_cols = ["span1", "span2", "target"]
        for model_name in self.model_names:
            table_cols += [f"pred: {model_name}", f"acc: {model_name}"]
        return table_cols

    def get_val_to_label_map(self, dropdown_color_option: str) -> Dict[float, str]:
        """Returns dict mapping t-SNE plot color option to a dictionary of (val, label) k/v pairs, for categorical plotting."""
        return {0.0: "False/No", 0.5: "Ambiguous", 1.0: "True/Yes"}

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dropping repetitive tokens in the DataFrame."""
        new_df = df.sort_values("target", axis="rows", ascending=False)
        new_df = new_df.drop_duplicates(["aggr_layer_12_tsne_x", "aggr_layer_12_tsne_y"])
        new_df[self.tsne_displayed_options] = new_df[self.tsne_displayed_options].applymap(
            lambda x: x if (x == 0 or x == 1) else 0.5
        )

        return new_df

    def get_tsne_rows(
        self, saved_click: Dict[str, pd.DataFrame], model_selector_val: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get rows from the tsne dataframe based on click."""
        tsne_df = self.map_model_to_df[self.map_model_name_to_id[model_selector_val]]
        selected_sentence = saved_click["sentence"]  # Getting sentence idx
        selected_rows = tsne_df.merge(  # Getting rows in df corresponding to selected sentence
            selected_sentence, on=["sentence"]
        )
        other_rows = tsne_df[tsne_df["id"] != selected_sentence.iloc[0]]
        return other_rows, selected_rows

    def get_specific_head_matrix(self, head_matrix_option: str, example_ids: List[int]) -> np.array:
        """Get summary plot matrix for a WSC-SPECIFIC summary options and example indices."""
        z = None
        for model_id, model_name in zip(self.model_ids, self.model_names):
            selected_attentions_spans = [
                np.stack(self.map_model_to_collateral[model_id][index]["attention_spans"])
                for index in example_ids
            ]  # to collect multiple attention matrices (from different examples)

            if head_matrix_option == f"{model_name}_coreference_intensity":
                z = np.mean(
                    [attn.max(axis=(2, 3)) for attn in selected_attentions_spans], axis=0
                )  # when more than 1 token in span> select max of them

            elif head_matrix_option == f"{model_name}_accuracy_based_on_head":
                z = self.map_model_to_collateral[model_id][example_ids[0]]["acc_matrix"]

            if self.mode == "compare":
                selected_attentions_spans1 = [
                    np.stack(self.map_model_to_collateral["model1"][index]["attention_spans"])
                    for index in example_ids
                ]  # to collect multiple attention matrices (from different examples)
                selected_attentions_spans2 = [
                    np.stack(self.map_model_to_collateral["model2"][index]["attention_spans"])
                    for index in example_ids
                ]  # to collect multiple attention matrices (from different examples)
                if head_matrix_option == "coreference_intensity_delta":
                    span_weights_delta = [
                        attentions_spans2.max(axis=(2, 3)) - attentions_spans1.max(axis=(2, 3))
                        for (attentions_spans1, attentions_spans2) in zip(
                            selected_attentions_spans1, selected_attentions_spans2
                        )
                    ]
                    z = np.mean(span_weights_delta, axis=0)
        return z

    def _unpad_tokens(self, attns: np.array, tokens: List[str]) -> Tuple[np.array, List[str]]:
        """Removing [PAD] tokens."""
        # Making sure we don't get any [PAD] tokens
        attns = attns[:, :, : len(tokens), : len(tokens)]
        # TODO: figure out why we only need this for superglue and not ABSA
        return attns, tokens

    def get_dist_per_layer(
        self, option: str, model_selector_val: str, selected_sentences: pd.DataFrame
    ) -> Tuple[np.array, np.array]:
        """Computing the mean distance per layer between span tokens for target/pred == 1 and target/pred == 0."""
        # Getting example rows and sentence rows (minus example rows) in df
        curr_model_full_df = self.map_model_to_full_df[
            self.map_model_name_to_id[model_selector_val]
        ]
        selected_sentence_rows = curr_model_full_df.loc[
            curr_model_full_df["sentence_idx"].isin(selected_sentences)
        ]
        all_ex_ids = selected_sentence_rows["id"].unique()

        span_agg_distance_0 = np.zeros(self.num_layers + 1)
        span_agg_distance_1 = np.zeros(self.num_layers + 1)
        count_0 = 0
        count_1 = 0

        for ex_id in all_ex_ids:
            ex_rows = selected_sentence_rows[selected_sentence_rows["id"] == ex_id]
            span1 = ex_rows["span1"].iloc[0]
            span2 = ex_rows["span2"].iloc[0]
            span1_rows = ex_rows[ex_rows["span"] == span1]
            span2_rows = ex_rows[ex_rows["span"] == span2]
            span1_coords = [
                np.array(
                    (
                        span1_rows[f"layer_{layer:02}_tsne_x"].mean(),
                        span1_rows[f"layer_{layer:02}_tsne_y"].mean(),
                    )
                )
                for layer in range(self.num_layers + 1)
            ]
            span2_coords = [
                np.array(
                    (
                        span2_rows[f"layer_{layer:02}_tsne_x"].mean(),
                        span1_rows[f"layer_{layer:02}_tsne_y"].mean(),
                    )
                )
                for layer in range(self.num_layers + 1)
            ]

            dist_per_layer = np.array(
                [
                    np.linalg.norm(span1_coord - span2_coord)
                    for span1_coord, span2_coord in zip(span1_coords, span2_coords)
                ]
            )

            if ex_rows[option].iloc[0] == 1:
                span_agg_distance_1 += dist_per_layer
                count_1 += 1
            else:
                span_agg_distance_0 += dist_per_layer
                count_0 += 1

        # Averaging by number of examples
        span_mean_distance_0 = span_agg_distance_0 / count_0
        span_mean_distance_1 = span_agg_distance_1 / count_1
        return span_mean_distance_0, span_mean_distance_1

    def get_summary_table(self, saved_click: pd.DataFrame) -> Dict[str, Union[str, int]]:
        """Updates summary table with values associated with the saved click."""
        selected_sentence = saved_click["sentence"]
        cols = ["span1", "span2", "acc", "pred", "target"]
        models_sentence_df = {}
        for model_id, model_name in zip(self.model_ids, self.model_names):
            sentence_df = self.map_model_to_df[model_id]
            sentence_df = sentence_df[sentence_df["sentence"] == selected_sentence]
            sentence_df = sentence_df[cols]
            sentence_df = sentence_df.drop_duplicates()
            sentence_df.columns = [
                "span1",
                "span2",
                f"acc: {model_name}",
                f"pred: {model_name}",
                f"target: {model_name}",
            ]
            models_sentence_df[model_id] = sentence_df

        if self.mode == "normal":
            sentence_df = models_sentence_df["model1"]
        elif self.mode == "compare":
            sentence_df = models_sentence_df["model1"].merge(
                models_sentence_df["model2"], on=["span1", "span2"]
            )
            sentence_df = sentence_df.drop(
                f"target: {self.map_model_id_to_name['model2']}", axis="columns"
            )
        sentence_df = sentence_df.rename(
            {f"target: {self.map_model_id_to_name['model1']}": "target"}, axis="columns"
        )
        cols = [
            "span1",
            "span2",
            "target",
            f"pred: {self.map_model_id_to_name['model1']}",
            f"acc: {self.map_model_id_to_name['model1']}",
        ]

        if self.mode == "compare":
            cols += [
                f"pred: {self.map_model_id_to_name['model2']}",
                f"acc: {self.map_model_id_to_name['model2']}",
            ]

        sentence_df = sentence_df[cols]
        return sentence_df.to_dict("records")


class ABSATask(Task):
    def __init__(self, *args, **kwargs):
        super(ABSATask, self).__init__(*args, **kwargs)
        self.name = "absa"
        self.df_col_to_plot = "aspect"

    def _init_tsne_layout(self) -> List[str]:
        """Get layout options for the tsne plot."""
        tsne_internal_options = ["domain"]
        for model_name in self.model_names:
            tsne_internal_options += [f"f1_{model_name}"]
        if self.mode == "compare":
            tsne_internal_options += ["f1_delta"]
        tsne_plot_description = "Plotting Aspect Tokens"
        tsne_displayed_options = tsne_internal_options
        return tsne_internal_options, tsne_displayed_options, tsne_plot_description

    def _add_head_matrix_options(self, head_matrix_options: List[str]) -> List[str]:
        """Add grammar correlation option for ABSA."""
        for model_name in self.model_names:
            head_matrix_options += [f"{model_name}_grammar_correlation"]
        if self.mode == "compare":
            head_matrix_options += ["grammar_correlation_delta"]

        return head_matrix_options

    def _init_table_cols_layout(self) -> List[str]:
        """Get column labels for the summary table for ABSA."""
        table_cols = ["words", "target"]
        for model_name in self.model_names:
            table_cols += [f"pred: {model_name}"]
        return table_cols

    def get_val_to_label_map(self, dropdown_color_option: str) -> Dict[float, str]:
        """Returns dict mapping t-SNE plot color option to a dictionary of (val, label) k/v pairs, for categorical plotting."""
        return (
            {v: k for k, v in self.domain_map.items()}
            if dropdown_color_option == "domain"
            else None
        )

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies model-specific preprocessing to a dataframe."""
        return df

    def _process_all_df(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Adds additional information to input DataFrames.

        Adds prediction and f1 information from model1_df to model2_df and
        vice versa. Additionally computes the f1 deltas.

        Args:
            model1_df: DataFrame for first model.
            model2_df: DataFrame for second model.

        Returns:
            The modified versions of both DataFrames.
        """
        if self.mode == "normal":
            (model1_df,) = dfs.values()
            model1_df = model1_df.rename(
                columns={
                    "f1": f"f1_{self.map_model_id_to_name['model1']}",
                    "pred": f"pred_{self.map_model_id_to_name['model1']}",
                }
            )
            domains = model1_df["domain"].unique()
            self.domain_map = {domain: val for val, domain in enumerate(domains)}
            model1_df["domain"] = model1_df["domain"].map(self.domain_map)
            new_df = {}
            new_df["model1"] = model1_df
        else:
            # TODO: Merge both DataFrames into one
            model1_df, model2_df = dfs.values()
            model1_df = model1_df.rename(
                columns={
                    "f1": f"f1_{self.map_model_id_to_name['model1']}",
                    "pred": f"pred_{self.map_model_id_to_name['model1']}",
                }
            )
            model2_df = model2_df.rename(
                columns={
                    "f1": f"f1_{self.map_model_id_to_name['model2']}",
                    "pred": f"pred_{self.map_model_id_to_name['model2']}",
                }
            )

            # Need aspect key for merge for case where aspect get truncated (problem when using pivot phrase due to 64 max seq length truncation)
            model1_full_df = model1_df.merge(
                model2_df[
                    [
                        "aspect",
                        "id",
                        f"f1_{self.map_model_id_to_name['model2']}",
                        f"pred_{self.map_model_id_to_name['model2']}",
                    ]
                ],
                on=["id", "aspect"],
                how="inner",
            )
            model2_full_df = model2_df.merge(
                model1_df[
                    [
                        "aspect",
                        "id",
                        f"f1_{self.map_model_id_to_name['model1']}",
                        f"pred_{self.map_model_id_to_name['model1']}",
                    ]
                ],
                on=["id", "aspect"],
                how="inner",
            )

            model1_full_df["f1_delta"] = (
                model1_full_df[f"f1_{self.map_model_id_to_name['model1']}"]
                - model1_full_df[f"f1_{self.map_model_id_to_name['model2']}"]
            )
            model2_full_df["f1_delta"] = (
                model2_full_df[f"f1_{self.map_model_id_to_name['model2']}"]
                - model2_full_df[f"f1_{self.map_model_id_to_name['model1']}"]
            )

            model1_full_df = model1_full_df.drop_duplicates()
            model2_full_df = model2_full_df.drop_duplicates()

            # Creating numeric mapping for domain labels in dataframe
            domains = pd.concat([model1_full_df["domain"], model2_full_df["domain"]]).unique()
            self.domain_map = {domain: val for val, domain in enumerate(domains)}
            model1_full_df["domain"] = model1_full_df["domain"].map(self.domain_map)
            model2_full_df["domain"] = model2_full_df["domain"].map(self.domain_map)

            new_df = {}
            new_df["model1"] = model1_full_df
            new_df["model2"] = model2_full_df

        return new_df

    def get_tsne_rows(
        self, saved_click: Dict[str, pd.DataFrame], model_selector_val: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get rows from the tsne dataframe based on click."""
        tsne_df = self.map_model_to_df[self.map_model_name_to_id[model_selector_val]]
        # Getting example matching id and aspect
        selected_ex = saved_click[["aspect", "id"]]
        selected_rows = tsne_df.merge(selected_ex, on=["aspect", "id"])
        other_rows = tsne_df
        return other_rows, selected_rows

    def get_specific_head_matrix(self, head_matrix_option: str, example_ids: List[int]) -> np.array:
        """Get summary plot matrix for a ABSA-SPECIFIC summary options and example indices."""
        z = None
        for model_id, model_name in zip(self.model_ids, self.model_names):
            grammar_correlations = [
                self.map_model_to_collateral[model_name][index]["grammar_matrices"]
                for index in example_ids
            ]
            if head_matrix_option == f"{model_name}_grammar_correlation":
                z = np.mean(grammar_correlations, axis=0)

        if self.mode == "compare":
            grammar_correlations1 = [
                self.map_model_to_collateral["model1"][index]["grammar_matrices"]
                for index in example_ids
            ]
            grammar_correlations2 = [
                self.map_model_to_collateral["model2"][index]["grammar_matrices"]
                for index in example_ids
            ]
            if head_matrix_option == "gramar_correlation_delta":
                grammar_correlations_delta = [
                    grammar_correlation2 - grammar_correlation1
                    for (grammar_correlation1, grammar_correlation2) in zip(
                        grammar_correlations1, grammar_correlations2
                    )
                ]
                z = np.mean(grammar_correlations_delta, axis=0)
        return z

    def get_summary_table(self, saved_click: pd.DataFrame) -> Dict[str, Union[str, int]]:
        """Updates summary table with values associated with the saved click."""
        if self.mode == "normal":
            cols = ["sentence", "target", f"pred_{self.map_model_id_to_name['model1']}"]
            sentence_df = pd.DataFrame([saved_click[el].split(" ") for el in cols]).T
            sentence_df.columns = [
                "words",
                "target",
                f"pred: {self.map_model_id_to_name['model1']}",
            ]
        elif self.mode == "compare":
            cols = [
                "sentence",
                "target",
                f"pred_{self.map_model_id_to_name['model1']}",
                f"pred_{self.map_model_id_to_name['model2']}",
            ]
            sentence_df = pd.DataFrame([saved_click[el].split(" ") for el in cols]).T
            sentence_df.columns = [
                "words",
                "target",
                f"pred: {self.map_model_id_to_name['model1']}",
                f"pred: {self.map_model_id_to_name['model2']}",
            ]

        return sentence_df.to_dict("records")


def get_task(
    task_name: str,
    num_layers: int,
    model_names: List[str],
    collaterals_paths: List[str],
    df_paths: List[str],
) -> Task:
    """Gets a task for the app."""
    if len(collaterals_paths) == 1:
        mode = "normal"
        assert len(collaterals_paths) == 1, "Provided wrong number of csv"
    elif len(collaterals_paths) == 2:
        mode = "compare"
        assert len(df_paths) == 2, "Provided wrong number of csv"

    print(f"Starting app in '{mode}' mode for '{task_name}'")

    if task_name == "absa":
        return ABSATask(mode, num_layers, model_names, collaterals_paths, df_paths)
    elif task_name == "wsc":
        return WSCTask(mode, num_layers, model_names, collaterals_paths, df_paths)
