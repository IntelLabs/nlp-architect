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
""" Module specifying Dash app UI layout

The main function here is the get_layout() function which returns
the Dash/HTML layout for InterpreT.
"""

import base64
import dash_table
import dash_core_components as dcc
import dash_html_components as html

intel_dark_blue = "#0168b5"
intel_light_blue = "#04c7fd"


def get_layout(Task):
    model_name_p = [
        html.P(f"{name}: {path}") for name, path in zip(Task.model_names, Task.collaterals_paths)
    ]
    # Default Configs
    logoConfig = dict(displaylogo=False, modeBarButtonsToRemove=["sendDataToCloud"])
    image_filename = "./assets/intel_ai_logo.jpg"
    encoded_image = base64.b64encode(open(image_filename, "rb").read())

    # STYLE CONFIGS
    label_style = {
        "font-family": "verdana",
        "font-size": "20px",
        "justify-content": "center",
        "padding_top": "5px",
        "padding_bottom": "2px",
        "font-weight": "bold",
    }
    small_label_style = {
        "font-family": "verdana",
        "font-size": "12px",
        "justify-content": "center",
        "padding_top": "5px",
        "padding_bottom": "2px",
        "font-weight": "bold",
    }

    # Conditional Options
    tsne_options = [
        {"label": Task.map_model_id_to_name["model1"], "value": Task.map_model_id_to_name["model1"]}
    ]
    if Task.mode == "compare":
        tsne_options += [
            {
                "label": Task.map_model_id_to_name["model2"],
                "value": Task.map_model_id_to_name["model2"],
            }
        ]

    # Actual layout
    layout = html.Div(
        [
            html.Div(id="intermediate-value", style={"display": "none"}),
            html.Div(id="save_clicked_point", style={"display": "none"}),
            html.Div(id="save_selected_points", style={"display": "none"}),
            html.Div(id="save_entered_id", style={"display": "none"}),
            html.Div(id="debug", style={"display": "none"}),
            html.Div(id="save_head_layer", style={"display": "none"}),
            html.Div(id="multiselection", style={"display": "none"}),
            html.Div(id="range_slider_copy", style={"display": "none"}),
            html.Div(
                [
                    # Intel logo
                    html.Div(
                        [
                            html.Img(
                                src=f"data:image/png;base64,{encoded_image.decode()}",
                                style={
                                    "display": "inline",
                                    "height": str(174 * 0.18) + "px",
                                    "width": str(600 * 0.18) + "px",
                                    "position": "relative",
                                    "padding-right": "30px",
                                    "vertical-align": "middle",
                                },
                            ),
                            html.H1(
                                "Interpre",
                                style={
                                    "font-size": "60px",
                                    "textAlign": "center",
                                    "display": "inline",
                                    "color": intel_dark_blue,
                                    "vertical-align": "middle",
                                },
                            ),
                            html.H1(
                                "T",
                                style={
                                    "font-size": "60px",
                                    "textAlign": "center",
                                    "display": "inline",
                                    "color": intel_light_blue,
                                    "vertical-align": "middle",
                                },
                            ),
                            html.Div(
                                children=[
                                    html.Span("An Interactive Visualization Tool for "),
                                    html.Strong("Interpre", style={"color": intel_dark_blue}),
                                    html.Span("ting "),
                                    html.Strong("T", style={"color": intel_light_blue}),
                                    html.Span("ransformers"),
                                ],
                                style={
                                    "textAlign": "center",
                                    "font-size": "30px",
                                },
                            ),
                        ],
                        style={"textAlign": "center", "margin": "auto"},
                    )
                ],
                className="row",
                style={
                    "text-align": "center",
                    "margin-top": "2%",
                    "margin-right": "2%",
                    "margin-bottom": "2%",
                    "margin-left": "2%",
                },
            ),
            # TSNE stuff (left column)
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Selected Sentence", style=label_style),
                            html.Div(
                                id="sentence",
                                style={
                                    "margin-top": "40",
                                    "width": "100%",
                                    "height": "auto",
                                    "margin-bottom": "40",
                                },
                            ),
                        ],
                        style={
                            "width": "100%",
                            "float": "left",
                            "padding-bottom": "40px",
                            "text-align": "center",
                        },
                    ),
                    html.Div(
                        [
                            html.Label("Summary Table", style=label_style),
                            dash_table.DataTable(
                                id="table",
                                columns=[
                                    {"name": j, "id": i}
                                    for i, j in zip(Task.table_cols, Task.table_cols)
                                ],
                                data=[],
                                style_data={
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                    "width": "auto",
                                    "lineHeight": "15px",
                                },
                            ),
                        ],
                        className="five columns",
                        style={"width": "100%", "padding": "40px", "text-align": "center"},
                    )
                    if Task.name == "wsc"
                    else None,
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("t-SNE Embeddings", style=label_style),
                                    html.Div(
                                        [
                                            html.Label("Model", style=small_label_style),
                                            dcc.Dropdown(
                                                id="model_selector",
                                                options=tsne_options,
                                                value=Task.map_model_id_to_name["model1"],
                                            ),
                                        ],
                                        className="four columns",
                                        style={
                                            "margin-top": "40",
                                            "width": "49%",
                                            "text-align": "center",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Coloring", style=small_label_style),
                                            dcc.Dropdown(
                                                id="tsne_plot_options",
                                                options=[
                                                    {"label": el, "value": el}
                                                    for el in Task.tsne_displayed_options
                                                ],
                                                value=Task.tsne_displayed_options[0],
                                            ),
                                        ],
                                        className="four columns",
                                        style={
                                            "margin-top": "40",
                                            "width": "49%",
                                            "text-align": "center",
                                            "float": "right",
                                        },
                                    ),
                                ],
                                style={"padding": "20px", "text-align": "center"},
                            ),
                            html.Div(
                                [
                                    html.Label("Layer Selector", style=small_label_style),
                                    dcc.Slider(
                                        id="layer_slider",
                                        min=0,
                                        max=12,
                                        step=1,
                                        marks={i: i for i in range(0, 12 + 1, 1)},
                                        value=12,
                                    ),
                                ],
                                className="four columns",
                                style={
                                    "margin-top": "40",
                                    "width": "100%",
                                    "text-align": "center",
                                    "padding-top": "10px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        children=Task.tsne_plot_description,
                                        style={
                                            "textAlign": "center",
                                        },
                                    ),
                                    dcc.Graph(id="tsneMap", config=logoConfig),
                                ],
                                className="six columns",
                                style={"height": "100%", "width": "95%", "text-align": "center"},
                            ),
                        ],
                        className="two columns",
                        style={"width": "100%"},
                    ),
                    html.Div(
                        [
                            html.Label("Summary Table", style=label_style),
                            dash_table.DataTable(
                                id="table",
                                columns=[
                                    {"name": j, "id": i}
                                    for i, j in zip(Task.table_cols, Task.table_cols)
                                ],
                                data=[],
                                style_data={
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                    "width": "auto",
                                    "lineHeight": "15px",
                                },
                            ),
                        ],
                        className="five columns",
                        style={"width": "100%", "padding": "40px", "text-align": "center"},
                    )
                    if Task.name == "absa"
                    else None,
                    html.Div(
                        [
                            html.Label("Average t-SNE Distance Per Layer", style=label_style),
                            dcc.Dropdown(
                                id="layer_dist_plot_options",
                                options=[{"label": el, "value": el} for el in ["target", "pred"]],
                                value="pred",
                            ),
                            dcc.Graph(id="layer_dist_plot", config=logoConfig),
                        ],
                        className="six columns",
                        style={
                            "padding-top": "40",
                            "margin-top": "40",
                            "width": "100%",
                            "text-align": "center",
                        },
                    )
                    if Task.name == "wsc"
                    else None,
                ],
                className="row",
                style={"width": "45%", "float": "left", "padding": "20px", "padding-left": "40px"},
            ),
            # Attention stuff (right column)
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Selected Models", style=label_style),
                                    html.Div(
                                        model_name_p,
                                        style={
                                            "margin-top": "40",
                                            "height": "auto",
                                            "margin-bottom": "40",
                                        },
                                    ),
                                ],
                                style={
                                    "width": "90%",
                                    "float": "left",
                                    "padding-bottom": "40px",
                                    "text-align": "center",
                                },
                            ),
                            html.Div(
                                [
                                    html.Label("Head Summary", style=label_style),
                                    html.Div(
                                        [
                                            html.Label("Metrics", style=small_label_style),
                                            dcc.Dropdown(
                                                id="head_matrix_options",
                                                options=[
                                                    {"label": el, "value": el}
                                                    for el in Task.head_matrix_options
                                                ],
                                                value=f"{Task.map_model_id_to_name['model1']}_std",
                                            ),
                                        ],
                                        className="four columns",
                                        style={
                                            "margin-top": "40",
                                            "width": "49%",
                                            "text-align": "center",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Colorscale Range", style=small_label_style),
                                            html.Div(
                                                [
                                                    dcc.RangeSlider(
                                                        id="range_slider",
                                                        min=0.0,
                                                        max=1.0,
                                                        step=0.05,
                                                        value=[0, 0.3],
                                                        marks={
                                                            0.0: "0",
                                                            0.2: "0.2",
                                                            0.4: "0.4",
                                                            0.6: "0.6",
                                                            0.8: "0.8",
                                                            1.0: "1",
                                                        },
                                                    ),
                                                    dcc.RadioItems(
                                                        id="auto_rescale",
                                                        options=[
                                                            {"label": i, "value": i}
                                                            for i in ["Auto", "Manual"]
                                                        ],
                                                        value="Auto",
                                                        labelStyle={"display": "inline-block"},
                                                        style={"font-size": "12px"},
                                                        inputStyle={"margin-left": "20px"},
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="four columns",
                                        style={
                                            "margin-top": "40",
                                            "width": "49%",
                                            "text-align": "center",
                                        },
                                    ),
                                ],
                                className="four columns",
                                style={
                                    "margin-top": "40",
                                    "width": "90%",
                                    "text-align": "center",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="head_matrix", config=logoConfig),
                                ],
                                className="four columns",
                                style={
                                    "width": "90%",
                                    "text-align": "center",
                                },
                            ),
                        ],
                        className="two columns",
                        style={"width": "90%"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Attention Matrix/Map", style=label_style),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Model", style=small_label_style),
                                                    dcc.Dropdown(
                                                        id="attn_map_options",
                                                        options=[
                                                            {"label": el, "value": el}
                                                            for el in Task.attn_map_options
                                                        ],
                                                        value=Task.map_model_id_to_name["model1"],
                                                    ),
                                                ],
                                                className="four columns",
                                                style={
                                                    "margin-top": "0",
                                                    "width": "49%",
                                                    "text-align": "center",
                                                },
                                            ),
                                            html.Div(
                                                [
                                                    html.Label("View", style=small_label_style),
                                                    dcc.Dropdown(
                                                        id="attn_map_toggle",
                                                        options=[
                                                            {"label": el, "value": el}
                                                            for el in ("map", "matrix")
                                                        ],
                                                        value="matrix",
                                                    ),
                                                ],
                                                className="four columns",
                                                style={
                                                    "width": "49%",
                                                    "text-align": "center",
                                                    "float": "right",
                                                },
                                            ),
                                        ],
                                        style={
                                            "width": "100%",
                                            "text-align": "center",
                                            "padding-bottom": "60px",
                                            "padding-top": "10px",
                                        },
                                    ),
                                    html.Div(
                                        dcc.Graph(id="attn_head", config=logoConfig),
                                        style={
                                            "display": "block",
                                            "margin-left": "auto",
                                            "margin-right": "auto",
                                        },
                                    ),
                                ],
                                className="four columns",
                                style={
                                    "margin-top": "40",
                                    "padding": "20px",
                                    "width": "90%",
                                    "text-align": "center",
                                },
                            ),
                        ],
                        className="five columns",
                        style={
                            "margin-top": "40",
                            "text-align": "center",
                            "width": "90%",
                        },
                    ),
                ],
                className="row",
                style={"width": "45%", "float": "right", "padding": "20px"},
            ),
        ],
        style={"margin-left": "auto", "margin-right": "auto"},
    )

    return layout
