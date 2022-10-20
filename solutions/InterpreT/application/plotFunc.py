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
""" Module containing plotting functions.

These plotting functions are used in appConfiguration.py to
generate all the the plots for the UI.
"""

import base64
import copy
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

layout = dict(
    autosize=True,
    font=dict(color="black"),
    titlefont=dict(color="black", size=14),
    margin=dict(l=35, r=35, b=35, t=45),
    hovermode="closest",
    legend=dict(font=dict(size=10), orientation="h"),
)


def plot_head_matrix(z, head_layer, z_max, z_min):
    data = [
        dict(type="heatmap", z=z, colorscale="RdBu", reversescale=False, zmin=z_min, zmax=z_max),
        dict(
            type="scatter",
            x=[int(head_layer["head_no"])],
            y=[int(head_layer["layer_no"])],
            marker=dict(
                color="black",
                symbol="x",
                size=17,
                opacity=0.4,
            ),
        ),
    ]
    num_layers, num_heads = np.shape(z)
    layout_ind = copy.deepcopy(layout)
    layout_ind.update({"yaxis": dict(autorange="reversed")})
    layout_ind["yaxis"].update(
        {
            "title": "layer_no",
            "tickmode": "array",
            "ticktext": list(range(1, num_layers + 1)),
            "tickvals": list(range(num_layers)),
        }
    )
    layout_ind["xaxis"] = {
        "title": "head_no",
        "tickmode": "array",
        "ticktext": list(range(1, num_heads + 1)),
        "tickvals": list(range(num_heads)),
    }
    layout_ind.update({"margin": {"t": 20}})

    figure = dict(data=data, layout=layout_ind)

    return figure


def plot_tsne(
    token_evaluated,
    tsne_table,
    color_summary_val,
    layer_slider_val,
    selected_rows,
    color_summary_val_to_col,
    val_to_label,
):
    x = f"layer_{layer_slider_val:02}_tsne_x"
    y = f"layer_{layer_slider_val:02}_tsne_y"

    figure = go.Figure()

    # Continuous variable plotting
    if val_to_label is None:
        figure.add_trace(
            go.Scatter(
                x=tsne_table[x],
                y=tsne_table[y],
                hovertext=tsne_table[token_evaluated].astype("str"),
                mode="markers",
                marker=dict(
                    color=tsne_table[color_summary_val_to_col[color_summary_val]],
                    colorscale="Jet",
                    showscale=True,
                    cmin=-1.0 if color_summary_val.find("delta") != -1 else 0.0,
                    cmax=1.0,
                ),
                name=color_summary_val,
                showlegend=True,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=selected_rows[x],
                y=selected_rows[y],
                mode="markers+text",
                hovertext=selected_rows[token_evaluated].astype("str"),
                text=selected_rows[token_evaluated].astype("str"),
                textposition="top center",
                textfont=dict(size=15),
                marker=dict(
                    color="orange",
                    colorscale="Jet",
                    showscale=False,
                    symbol="star",
                    size=10,
                    opacity=1,
                    cmin=-1.0 if color_summary_val.find("delta") != -1 else 0.0,
                    cmax=1.0,
                ),
                showlegend=False,
            )
        )
        figure.update_layout(legend=dict(font=dict(size=15), y=1.1, x=0.9))

    # Discrete/categorical plotting
    else:
        for val in sorted(
            tsne_table[color_summary_val_to_col[color_summary_val]].unique(), reverse=True
        ):
            curr_trace_df = tsne_table[
                tsne_table[color_summary_val_to_col[color_summary_val]] == val
            ]
            figure.add_trace(
                go.Scatter(
                    x=curr_trace_df[x],
                    y=curr_trace_df[y],
                    hovertext=curr_trace_df[token_evaluated].astype("str"),
                    mode="markers",
                    name=val_to_label[val],
                )
            )

        figure.add_trace(
            go.Scatter(
                x=selected_rows[x],
                y=selected_rows[y],
                hovertext=selected_rows[token_evaluated].astype("str"),
                mode="markers+text",
                text=selected_rows[token_evaluated].astype("str"),
                textposition="top center",
                textfont=dict(size=15),
                marker=dict(
                    color="orange",
                    symbol="star",
                    size=10,
                ),
                name="Selected Sentence",
            )
        )
        figure.update_layout(legend_x=0, legend_y=1)

    figure.update_layout(margin={"l": 0, "r": 0, "b": 60, "t": 20})
    return figure


def plot_attn_matrix(tokens, attn):
    data = [
        dict(type="heatmap", z=attn),
    ]
    layout_ind = copy.deepcopy(layout)
    layout_ind.update(
        {"yaxis": dict(tickmode="array", tickvals=list(range(len(tokens))), ticktext=tokens)}
    )
    layout_ind.update(
        {
            "xaxis": dict(
                tickmode="array", tickvals=list(range(len(tokens))), ticktext=tokens, tickangle=45
            )
        }
    )
    layout_ind.update({"margin": {"t": 20}})
    figure = dict(data=data, layout=layout_ind)

    return figure


def plot_attn_map(tokens, attn, disabled_tokens=[]):
    # visual parameters
    width = 9
    word_height = 1
    pad = 0.1

    """ Expecting attn.shape == (len(tokens), len(tokens)) """
    n_tokens = len(tokens)
    assert attn.shape == (
        n_tokens,
        n_tokens,
    ), "attention shape must correspond to num of tokens"
    # disable special tokens
    if disabled_tokens:
        # recalculate attention scores without disabled_tokens
        disabled_ids = [tokens.index(tok) for tok in tokens if tok in disabled_tokens]
        # zero all attentions of disbaled_tokens
        for disable_tok_id in disabled_ids:
            attn[disable_tok_id] = 0  # zero row  (attention from)
            attn[:, disable_tok_id] = 0  # zero column (attention to)

    # plot
    fig, ax1 = plt.subplots(1, 1, figsize=(6.5, n_tokens / 4.5))
    fig.tight_layout()
    plt.subplots_adjust(left=0.3, bottom=0.0, right=0.7, top=1.0)

    ax1.axis("off")
    yoffset = 1
    xoffset = 0
    for position, word in enumerate(tokens):
        ax1.text(xoffset + 0, yoffset - position * word_height, word, ha="right", va="center")
        ax1.text(
            xoffset + width,
            yoffset - position * word_height,
            word,
            ha="left",
            va="center",
        )
    for i in range(0, n_tokens):
        for j in range(0, n_tokens):
            ax1.plot(
                [xoffset + pad, xoffset + width - pad],
                [yoffset - word_height * i, yoffset - word_height * j],
                color="blue",
                linewidth=2,
                alpha=float(attn[i, j]),
            )

    img_width, img_height = fig.get_size_inches() * fig.dpi  # size in pixels
    out_url = fig_to_uri(fig)

    # Plotly hack to set image as background image of plot
    fig = go.Figure()
    scale_factor = 1.0

    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0,
        )
    )

    # Configure axes
    fig.update_xaxes(visible=False, range=[0, img_width * scale_factor])

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x",
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=out_url,
        )
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig


def fig_to_uri(in_fig, close_all=True, **save_args):
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format="png", **save_args)
    if close_all:
        in_fig.clf()
        plt.close("all")
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def plot_layer_dist(dist_ones, dist_zeros, option):
    layout_ind = copy.deepcopy(layout)
    layout_ind["title"] = f"Average distance between span tokens, grouped by {option}, per layer"
    layout_ind["uirevision"] = "some_constant"
    layout_ind["yaxis"] = {"title": "Average Distance Between Spans (t-SNE)"}
    layout_ind["xaxis"] = {"title": "Layer"}
    layout_ind["showlegend"] = True

    x = list(range(len(dist_ones)))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=dist_ones,
            mode="lines",
            name="True Coreferent Spans"
            if option == "target"
            else "Predicted to be Coreferent Spans",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=dist_zeros,
            mode="lines",
            name="False Coreferent Spans"
            if option == "target"
            else "Predicted NOT to be Coreferent Spans",
        )
    )
    figure.add_trace(
        go.Scatter(x=x, y=dist_zeros - dist_ones, mode="lines", name="Delta (Red - Blue)")
    )

    figure.layout = layout_ind
    figure.update_layout(legend=dict(font=dict(size=15), y=-0.2))

    return figure
