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
""" Module containing the main Dash app code.

The main function in this module is configureApp() which starts
the Dash app on a Flask server and configures it with callbacks.
"""

import os

import dash
import pandas as pd
from flask import Flask
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from termcolor import cprint

import appLayout
import plotFunc


def configureApp(Task):
    app = start_app(Task)

    @app.callback(Output("save_head_layer", "children"), [Input("head_matrix", "clickData")])
    def save_head_matrix_click(click_data):
        if click_data is None:
            return_dict = dict(head_no="6", layer_no="9")
        else:
            x, y = get_click_coords(click_data)
            return_dict = dict(head_no=x, layer_no=y)
        return return_dict

    @app.callback(
        [Output("head_matrix", "figure"), Output("range_slider_copy", "value")],
        [
            Input("save_clicked_point", "children"),
            Input("save_selected_points", "children"),
            Input("head_matrix_options", "value"),
            Input("save_head_layer", "children"),
            Input("auto_rescale", "value"),
            Input("range_slider", "value"),
        ],
        [State("multiselection", "children")],
    )
    def plot_head_summary(
        saved_click,
        saved_points,
        head_matrix_option,
        head_layer,
        auto_rescale,
        range_slider,
        multiselection,
    ):
        ctx = dash.callback_context
        input_trigger = get_input_trigger(ctx)
        if input_trigger == "save_selected_points" or multiselection:
            saved_points = pd.read_json(saved_points, orient="split")
            indices = saved_points["id"]
        else:
            saved_click = pd.read_json(saved_click, orient="split").iloc[0]
            indices = [saved_click["id"]]

        z = Task.get_head_matrix(head_matrix_option, indices)
        if auto_rescale == "Manual":
            z_max = range_slider[1]
            z_min = range_slider[0]
        else:
            z_max = z.max()
            z_min = z.min()
            range_slider = [z_min, z_max]
        fig = plotFunc.plot_head_matrix(z, head_layer, z_max, z_min)

        return fig, range_slider

    @app.callback(
        Output("range_slider", "value"),
        [Input("range_slider_copy", "value")],
    )
    def copy_range_slider_value(range_slider):
        return range_slider

    @app.callback(
        Output("attn_head", "figure"),
        [
            Input("save_clicked_point", "children"),
            Input("save_head_layer", "children"),
            Input("attn_map_options", "value"),
            Input("attn_map_toggle", "value"),
        ],
    )
    def plot_attn_map(saved_click, head_layer, attn_map_option, attn_map_toggle):
        saved_click = pd.read_json(saved_click, orient="split").iloc[0]

        layer = int(head_layer["layer_no"])
        head = int(head_layer["head_no"])

        example_id = saved_click["id"]
        attns, tokens = Task.get_attn_map(attn_map_option, example_id)

        head_id = attns[layer][head]
        disabled_tokens = ["[CLS]", "[SEP]"]

        if attn_map_toggle == "map":
            src = plotFunc.plot_attn_map(tokens, head_id, disabled_tokens)
        elif attn_map_toggle == "matrix":
            src = plotFunc.plot_attn_matrix(tokens, head_id)

        return src

    @app.callback(
        Output("tsneMap", "figure"),
        [
            Input("tsne_plot_options", "value"),
            Input("layer_slider", "value"),
            Input("save_clicked_point", "children"),
            Input("model_selector", "value"),
        ],
    )
    def plot_tsne(dropdown_color_option, layer_slider_val, saved_click, model_selector_val):
        saved_click = pd.read_json(saved_click, orient="split")
        other_rows, selected_rows = Task.get_tsne_rows(saved_click, model_selector_val)
        df_column_to_plot = Task.get_df_col_to_plot()
        dropdown_to_df_col_map = (
            Task.get_dropdown_to_df_col_map()
        )  # Maps dropdown options to dataframe columns
        val_to_label = Task.get_val_to_label_map(
            dropdown_color_option
        )  # Maps values for the column to labels based on dropdown_color_option
        figure = plotFunc.plot_tsne(
            df_column_to_plot,
            other_rows,
            dropdown_color_option,
            layer_slider_val,
            selected_rows,
            dropdown_to_df_col_map,
            val_to_label,
        )
        return figure

    @app.callback(
        Output("save_clicked_point", "children"),
        [Input("tsneMap", "clickData")],
        [State("layer_slider", "value"), State("model_selector", "value")],
    )
    def save_tsne_map_click(click_data, layer_slider_val, model_selector_val):
        if click_data is None:
            # Default row to use on startup
            df_temp = Task.map_model_to_df["model1"]
            # Selecting as default the example of the paper
            selected_row = (
                df_temp[df_temp["sentence"].str.contains("got back")].head(1)
                if Task.get_name() == "wsc"
                else df_temp.head(1)
            )
        else:
            # Querying row based on tsne map click
            selected_point = pd.DataFrame(click_data["points"])[["x", "y"]]
            x_coord = selected_point["x"].iloc[0]
            y_coord = selected_point["y"].iloc[0]
            model_id = Task.map_model_name_to_id[model_selector_val]
            curr_model_df = Task.map_model_to_df[model_id]
            selected_row = curr_model_df[
                (curr_model_df[f"layer_{layer_slider_val:02}_tsne_x"] == x_coord)
                & (curr_model_df[f"layer_{layer_slider_val:02}_tsne_y"] == y_coord)
            ]

        # Saving row corresponding to clicked aspect
        return selected_row.to_json(orient="split")

    @app.callback(
        Output("save_selected_points", "children"),
        [Input("tsneMap", "selectedData")],
        [State("layer_slider", "value"), State("model_selector", "value")],
        prevent_initial_call=True,
    )
    def save_tsne_map_selection(selected_data, layer_slider_val, model_selector_val):
        if selected_data is None:
            # Don't do anything on startup
            raise PreventUpdate
        else:
            # Querying rows based on tsne map selection
            selected_points = pd.DataFrame(selected_data["points"])[["x", "y"]]
            model_id = Task.map_model_name_to_id[model_selector_val]
            selected_rows = Task.map_model_to_df[model_id].merge(
                selected_points,
                left_on=[
                    f"layer_{layer_slider_val:02}_tsne_x",
                    f"layer_{layer_slider_val:02}_tsne_y",
                ],
                right_on=["x", "y"],
            )

            # Saving row corresponding to clicked aspect
        return selected_rows.to_json(orient="split")

    @app.callback(
        Output("multiselection", "children"),
        [
            Input("tsneMap", "clickData"),
            Input("tsneMap", "selectedData"),
        ],
    )
    def set_multiselection_bool(click_data, selected_data):
        if click_data is None:
            # Default to false on startup
            return False

        # Checking trigger
        ctx = dash.callback_context
        input_trigger = get_input_trigger_full(ctx)
        if input_trigger == "tsneMap.selectedData":
            return True
        else:
            return False

    @app.callback(Output("sentence", "children"), [Input("save_clicked_point", "children")])
    def display_sentence(saved_click):
        saved_click = pd.read_json(saved_click, orient="split")
        return saved_click["sentence"]

    @app.callback(Output("table", "data"), [Input("save_clicked_point", "children")])
    def update_summary_table(saved_click):
        saved_click = pd.read_json(saved_click, orient="split").iloc[0]
        return_list = Task.get_summary_table(saved_click)
        return return_list

    if Task.get_name() == "wsc":

        @app.callback(
            Output("layer_dist_plot", "figure"),
            [
                Input("save_clicked_point", "children"),
                Input("save_selected_points", "children"),
                Input("layer_dist_plot_options", "value"),
                Input("model_selector", "value"),
            ],
            [State("multiselection", "children")],
        )
        def plot_dist_per_layer(
            saved_click, saved_points, option, model_selector_val, multiselection
        ):
            ctx = dash.callback_context
            input_trigger = get_input_trigger(ctx)
            if input_trigger == "save_selected_points" or multiselection:
                saved_points = pd.read_json(saved_points, orient="split")
                selected_sentences = saved_points["sentence_idx"].unique()
            else:
                saved_click = pd.read_json(saved_click, orient="split")
                selected_sentences = saved_click[
                    "sentence_idx"
                ].unique()  # Should be a single number

            span_mean_distance_0, span_mean_distance_1 = Task.get_dist_per_layer(
                option, model_selector_val, selected_sentences
            )

            figure = plotFunc.plot_layer_dist(span_mean_distance_1, span_mean_distance_0, option)
            return figure

    return app


def start_app(Task):
    print("Starting server")
    server = Flask(__name__)
    server.secret_key = os.environ.get("secret_key", "secret")

    app = dash.Dash(__name__, server=server, url_base_pathname="/")  # noqa: E501
    app.title = "InterpreT"
    app.layout = appLayout.get_layout(Task)
    return app


def printPageLink(hostname, port):

    print("\n\n")
    cprint("------------------------" "------------------------", "green")
    cprint("App Launched!", "red")
    cprint("Link (Use Chrome):", "red")

    pageLink = "http://{}.to.intel.com:{}/".format(hostname, port)
    cprint(pageLink, "blue")
    cprint("------------------------" "------------------------", "green")


def get_click_coords(click_data):
    return click_data["points"][0]["x"], click_data["points"][0]["y"]


def get_input_trigger(ctx):
    return ctx.triggered[0]["prop_id"].split(".")[0]


def get_input_trigger_full(ctx):
    return ctx.triggered[0]["prop_id"]
