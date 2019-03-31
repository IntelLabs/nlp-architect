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
# pylint: disable=global-statement
import csv
import logging
import sys
import time
from os import path
from os.path import isfile

import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, layout, WidgetBox
from bokeh.models import ColumnDataSource, Div, Row, ranges, LabelSet, CDSView, IndexFilter
from bokeh.models.widgets import DataTable, TableColumn, RadioGroup, Dropdown, Tabs, Panel, \
    TextInput, Button, Slider
from bokeh.plotting import figure

from nlp_architect.solutions.trend_analysis.trend_analysis import analyze
from nlp_architect import LIBRARY_OUT

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
dir = str(LIBRARY_OUT / 'trend-analysis-data')
graph_data_path = path.join(dir, 'graph_data.csv')
filter_data_path = path.join(dir, 'filter_phrases.csv')
target_scores_path = path.join(dir, 'target_scores.csv')
ref_scores_path = path.join(dir, 'ref_scores.csv')
max_len = 50
gd = None
top_before = None
top_after = None
hot_trends = None
cold_trends = None
trend_clustering = None
custom_trends = None
topic_clustering_before = None
topic_clustering_after = None
white_list = []
custom_nps = []
ref_data_name = ""
target_data_name = ""
apply_filter = False
dont_regenerate = False
filter_item = -1
all_topics = []
filter_rows = []
table_columns = [
    TableColumn(field="topics", title="Topics")
]


# create ui components

filter_topics_table_source = ColumnDataSource(data={})
view1 = CDSView(source=filter_topics_table_source, filters=[IndexFilter()])
filter_topics_table = DataTable(
    source=filter_topics_table_source, view=view1, columns=table_columns, width=500,
    selectable=True, scroll_to_selection=True, css_classes=['filter_topics_table'])
filter_custom_table_source = ColumnDataSource(data={})
filter_custom_table = DataTable(
    source=filter_custom_table_source, columns=table_columns, width=500, selectable=True,
    css_classes=['filter_custom_table'])
radio_group_area = RadioGroup(
    labels=["All", "Top Topics", "Trends", "Trend Clustering", "Custom Trends", "Topic Clustering",
            "Filter"], active=0, inline=True, width=400)
n_menu = [("5", "5"), ("10", "10"), ("20", "20"), ("30", "30")]
n_clusters_menu = [("20", "20"), ("50", "50"), ("100", "100"), ("300", "300"), ("500", "500")]
scores = [("0", "0"), ("0.1", "0.1"), ("0.25", "0.25"), ("0.5", "0.5"), ("1", "1")]
top_n_dropdown = Dropdown(label="Show Top 10 Phrases", button_type="warning",
                          menu=n_menu, value="10")
top_n_clusters_dropdown = Dropdown(label="Show Top 100 Clusters",
                                   button_type="warning", menu=n_clusters_menu, value="100")
tfidf_slider = Slider(start=0, end=1, value=0, step=.05, title="Informativeness")
cval_slider = Slider(start=0, end=1, value=0, step=.05, title="Completeness")
freq_slider = Slider(start=0, end=1, value=0, step=.05, title="Linguistical")
filter_topics_tab = Panel(child=filter_topics_table, title="Filter Topics")
filter_custom_tab = Panel(child=filter_custom_table, title="Custom Trends")
filter_tabs = Tabs(tabs=[filter_topics_tab, filter_custom_tab])
search_input_box = TextInput(title="Search:", value="", width=300)
search_button = Button(label="Go", button_type="success", width=50, css_classes=['search_button'])
clear_button = Button(label="Clear", button_type="success", width=50, css_classes=['clear_button'])
buttons_layout = column(Div(height=0), Row(search_button, clear_button))
analyse_button = Button(label="re-analyze", button_type="success")
filter_label = Div(
    text="", style={'color': 'red', 'padding-bottom': '0px', 'font-size': '15px'})
label = Div(
    text="", style={'color': 'red', 'padding-bottom': '0px', 'font-size': '15px'})
config_box = WidgetBox(tfidf_slider, cval_slider, freq_slider)
graphs_area = column(children=[])


# define page layout

config_area = layout(
    [
        [radio_group_area],
        [top_n_dropdown],
        [top_n_clusters_dropdown],
        [config_box],
        [analyse_button],


    ])
grid = layout(
    [
        [Div(width=500), Div(text="<H1>AIPG Trend Analysis</H1>")],
        [label],
        [config_area, Div(width=50), graphs_area]
    ]
)


# define functions

def reset_filters():
    """
    Resets the filter tables (e.g. after clearing search)
    """
    logger.info("reset filters")
    global filter_item
    filter_item = -1
    filter_topics_table.view.filters = [IndexFilter()]
    filter_custom_table.view.filters = [IndexFilter()]
    filter_label.text = ""


def refresh_filter_area():
    graphs_area.children = [Row(search_input_box, buttons_layout), filter_label,
                            filter_tabs]


def get_valid_phrases():
    """
    Get a list of all whitelist/valid phrases (which were not filtered-out by the user)
    """
    return [x[0] for x in all_topics if x[1] == '1']


def get_custom_phrases():
    """
    Get a list of all phrases selected by the user for the custom trends graph
    """
    return [x[0] for x in all_topics if x[2] == '1']


def get_valid_indices():
    """
    Get the filter-table indices of all whitelist/valid phrases
    """
    return [i for i, val in enumerate(all_topics) if val[1] == '1']


def is_valid_topic(index):
    """
    Check if a certain phrase should be presented
    """
    return all_topics[index][1] == '1'


def get_custom_indices():
    """
    Get the custom-table indices of all custom selected phrases
    """
    return [i for i, val in enumerate(all_topics) if val[2] == '1']


def search(item):
    """
    Search an item in the filter tables
    """
    row = -1
    try:
        row = filter_rows.index(item)
    except Exception:
        logger.warning("%s was not found", str(item))
    return row


def cut_phrase(phrase):
    """
    Cut phrases which exceeds phrase max length to present
    """
    return phrase if len(phrase) < max_len else phrase[:max_len] + "..."


def read_filter_graph_data():
    global all_topics, filter_rows
    all_topics = []
    filter_rows = []
    with open(filter_data_path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in enumerate(reader):
            all_topics.append(line[1])
            filter_rows.append(line[1][0])


def create_topic_clustering_before_graph(top_n):
    if 'Trend Clustering' not in gd['reports'].values:  # no clustering data
        return None
    header = gd['reports'][6]
    scatter_series = gd[(gd['x_ref'] != -1) & (gd['x_ref'].notna())]
    phrases = scatter_series['ref_topic'].values
    x_b = scatter_series['x_ref'].values
    y_b = scatter_series['y_ref'].values
    valid_phrases, valid_x, valid_y = [], [], []
    ctr = 0
    for i in range(len(phrases)):
        if ctr == top_n:
            break
        phrase = phrases[i]
        if phrase in white_list:
            valid_phrases.append(phrase)
            valid_x.append(x_b[i])
            valid_y.append(y_b[i])
            ctr += 1

    source = ColumnDataSource(data=dict(
        x=valid_x,
        y=valid_y,
        topics=valid_phrases
    ))

    p = figure(title=header, plot_width=600, plot_height=400, tooltips='@topics')
    p.circle(x='x', y='y', size=13, color="#335fa5",
             alpha=1, source=source)
    return p


def create_topic_clustering_after_graph(top_n):
    if 'Trend Clustering' not in gd['reports'].values:  # no clustering data
        return None
    header = gd['reports'][7]
    scatter_series = gd[(gd['x_tar'] != -1) & (gd['x_tar'].notna())]
    phrases = scatter_series['tar_topic'].values
    x_a = scatter_series['x_tar'].values
    y_a = scatter_series['y_tar'].values
    valid_phrases, valid_x, valid_y = [], [], []
    ctr = 0
    for i in range(len(phrases)):
        if ctr == top_n:
            break
        phrase = phrases[i]
        if phrase in white_list:
            valid_phrases.append(phrase)
            valid_x.append(x_a[i])
            valid_y.append(y_a[i])
            ctr += 1

    source = ColumnDataSource(data=dict(
        x=valid_x,
        y=valid_y,
        topics=valid_phrases
    ))

    p = figure(title=header, plot_width=600, plot_height=400, tooltips='@topics')
    p.circle(x='x', y='y', size=13, color="#335fa5",
             alpha=1, source=source)

    return p


def regenerate_graphs(top_n_phrases, top_n_clusters):
    logger.info("regenerate graphs")
    global top_before, top_after, hot_trends, cold_trends, trend_clustering, \
        custom_trends, white_list, custom_nps, topic_clustering_before, topic_clustering_after
    white_list = get_valid_phrases()
    custom_nps = get_custom_phrases()
    top_before = create_top_before_graph(int(top_n_phrases))
    top_after = create_top_after_graph(int(top_n_phrases))
    hot_trends = create_trend_graphs(int(top_n_phrases))[0]
    cold_trends = create_trend_graphs(int(top_n_phrases))[1]
    if top_n_clusters != 0:  # no w2v model
        trend_clustering = create_trend_clustering_graph(int(top_n_clusters))
        topic_clustering_before = create_topic_clustering_before_graph(
            int(top_n_clusters))
        topic_clustering_after = create_topic_clustering_after_graph(
            int(top_n_clusters))
    custom_trends = create_custom_trends_graph()


def create_graphs(top_n_phrases, top_n_clusters):
    logger.info("create graphs")
    global topic_clustering_before, topic_clustering_after, gd
    if gd is None:  # initialization
        gd = pd.read_csv(graph_data_path)
        read_filter_graph_data()  # read only once- then update using the filter table
        global white_list, custom_nps
        white_list = get_valid_phrases()
        custom_nps = get_custom_phrases()
        set_weights()
    regenerate_graphs(top_n_phrases, top_n_clusters)


def set_weights():
    weights_series = gd['weights']
    tfidf_slider.value = weights_series[0]
    cval_slider.value = weights_series[1]
    freq_slider.value = weights_series[2]


def re_analyze():
    logger.info("re-analysing scores")
    label.text = "Re-Analysing... Please Wait..."
    global gd
    if isfile(target_scores_path) and isfile(ref_scores_path):
        analyze(target_scores_path, ref_scores_path, tar_header=target_data_name,
                ref_header=ref_data_name,
                re_analysis=True, tfidf_w=float(tfidf_slider.value),
                cval_w=float(cval_slider.value), lm_w=float(freq_slider.value))
        gd = None
        create_graphs(top_n_dropdown.value, int(top_n_clusters_dropdown.value))
        if radio_group_area.active == 0:
            handle_selected_graph(0)
        else:
            radio_group_area.active = 0
        label.text = "Done!"
        time.sleep(2)
        label.text = ""
    else:
        logger.info("scores files are missing")
        label.text = "Error: scores files are missing!"
        time.sleep(2)
    logger.info("re-analysing done")


def create_custom_trends_graph():
    header = gd['reports'][4]
    phrases = gd.values[:, 9]
    imp_change = gd.values[:, 10]
    phrases_h, changes_h = [], []
    phrases_c, changes_c = [], []
    text_h, text_c = [], []
    custom_count = 0
    for i in range(len(phrases)):
        phrase = phrases[i]
        change = float(imp_change[i])
        if phrase in custom_nps:
            if custom_count > 30:
                logger.warning("too many custom phrases (>30). Showing partial list")
                break
            phrase = cut_phrase(phrase)
            if change > 0:
                phrases_h.append(phrase)
                changes_h.append(change)
                text_h.append('+' + str(("%.1f" % change)) + '%')
            else:
                phrases_c.append(phrase)
                changes_c.append(change)
                text_c.append(str(("%.1f" % change)) + '%')
            custom_count += 1

    changes = changes_h + changes_c
    text = text_h + text_c
    trends = phrases_h + phrases_c
    colors = []

    if len(changes_h) > 0:
        for i in range(len(changes_h)):
            colors.append("#1d6d34")
    if len(changes_c) > 0:
        for i in range(len(changes_c)):
            colors.append("#912605")
    if len(changes) < 10:  # pad with 10 blanks
        changes += [0] * (10 - len(changes))
        text += ' ' * (10 - len(text))
        trends += [str((10 - len(trends)) - i) for i in range(0, (10 - len(trends)))]
        colors += ['white'] * (10 - len(colors))

    source = ColumnDataSource(
        dict(y=trends[::-1], x=changes[::-1], colors=colors[::-1], text=text[::-1]))

    plot = figure(title=header, plot_width=600, plot_height=400, tools="save",
                  x_range=ranges.Range1d(start=min(changes) - 10, end=max(changes) + 20),
                  y_range=source.data["y"], tooltips='@y<br>change: @x')
    labels = LabelSet(x=max(changes) + 5, y='y', text='text', level='glyph',
                      x_offset=0, y_offset=-10, source=source,
                      render_mode='canvas')

    plot.hbar(source=source, right='x', y='y', left=0, height=0.5, color='colors')
    plot.add_layout(labels)

    return plot


def create_trend_graphs(top_n):
    header_h = gd['reports'][2]
    header_c = gd['reports'][3]
    phrases = gd.values[:, 9]  # all trends
    imp_change = gd.values[:, 10]
    data_h, data_c = {}, {}
    text_h, text_c = [], []
    hot, cold = 0, 0  # counters for each list
    plot_h = figure(plot_width=600, plot_height=400, title=header_h)
    plot_c = figure(plot_width=600, plot_height=400, title=header_c)
    for i in range(len(phrases)):
        if hot == top_n and cold == top_n:
            break
        phrase = phrases[i]
        change = float(imp_change[i])
        if phrase in white_list:
            phrase = cut_phrase(phrase)
            if change > 0:
                if hot < top_n:
                    data_h[phrase] = change
                    text_h.append('+' + str(("%.1f" % change)) + '%')
                    hot += 1
            elif cold < top_n:
                data_c[phrase] = change
                text_c.append(str(("%.1f" % change)) + '%')
                cold += 1
    if len(data_h.keys()) > 0:
        phrases_h = sorted(data_h, key=data_h.get)
        changes_h = sorted(data_h.values())
        source = ColumnDataSource(
            dict(y=phrases_h, x=changes_h, text=text_h[::-1]))
        title = header_h
        plot_h = figure(plot_width=600, plot_height=400, tools="save", title=title,
                        x_range=ranges.Range1d(start=0, end=max(changes_h) + 20),
                        y_range=source.data["y"], tooltips='@y<br>change: @x')
        labels = LabelSet(x='x', y='y', text='text', level='glyph',
                          x_offset=5, y_offset=0, source=source,
                          render_mode='canvas', text_color="#1d6d34")
        plot_h.hbar(source=source, right='x', y='y', left=0, height=0.5, color="#1d6d34")
        plot_h.add_layout(labels)
        plot_h.xaxis.visible = False
    if len(data_c.keys()) > 0:
        phrases_c = sorted(data_c, key=data_c.get)
        changes_c = sorted(data_c.values())
        source = ColumnDataSource(
            dict(y=phrases_c[::-1], x=changes_c[::-1], text=text_c[::-1]))
        plot_c = figure(plot_width=600, plot_height=400, tools="save",
                        title=header_c,
                        x_range=ranges.Range1d(start=min(changes_c) - 10, end=20),
                        y_range=source.data["y"],
                        tooltips='@y<br>change: @x')
        labels = LabelSet(x=0, y='y', text='text', level='glyph',
                          x_offset=5, y_offset=0, source=source,
                          render_mode='canvas', text_color="#912605")
        plot_c.hbar(source=source, right='x', y='y', left=0, height=0.5,
                    color="#912605")
        plot_c.add_layout(labels)
        plot_c.xaxis.visible = False

    return [plot_h, plot_c]


def create_trend_clustering_graph(top_n):
    if 'Trend Clustering' not in gd['reports'].values:  # no clustering data
        return None
    header = gd['reports'][5]
    scatter_series_pos = gd[(gd['x_tre'] != -1) & (gd['x_tre'].notna()) & (gd['change'] > 0)]
    scatter_series_neg = gd[
        (gd['x_tre'] != -1) & (gd['x_tre'].notna()) & (gd['change'] < 0)]
    hot_phrases = scatter_series_pos['trends'].values
    cold_phrases = scatter_series_neg['trends'].values
    hot_change = scatter_series_pos['change'].values
    cold_change = scatter_series_neg['change'].values
    hot_x_t = scatter_series_pos['x_tre'].values
    cold_x_t = scatter_series_neg['x_tre'].values
    hot_y_t = scatter_series_pos['y_tre'].values
    cold_y_t = scatter_series_neg['y_tre'].values
    phrases_h, changes_h, x_h, y_h, hover_data_h = [], [], [], [], []
    phrases_c, changes_c, x_c, y_c, hover_data_c = [], [], [], [], []
    opacity_h, opacity_c = [], []
    hot, cold = 0, 0  # counters for each list
    if len(hot_change) > 0 and len(cold_change) > 0:
        highest_change = max(hot_change[0], abs(cold_change[0]))
        opacity_norm = 1 / (highest_change) if highest_change != 0 else 0.05

    for i in range(len(hot_phrases)):
        if hot == top_n:
            break
        phrase = hot_phrases[i]
        change = float(hot_change[i])
        if phrase in white_list:
            phrases_h.append(phrase)
            changes_h.append(change)
            x_h.append(hot_x_t[i])
            y_h.append(hot_y_t[i])
            hover_data_h.append(phrase)
            opacity = opacity_norm * abs(change)
            opacity_h.append(
                opacity if opacity > 0.05 else 0.05)
            hot += 1

    for j in range(len(cold_phrases)):
        if cold == top_n:
            break
        phrase = cold_phrases[j]
        change = float(cold_change[j])
        if phrase in white_list:
            phrases_c.append(phrase)
            changes_c.append(change)
            x_c.append(cold_x_t[j])
            y_c.append(cold_y_t[j])
            hover_data_c.append(phrase)
            opacity = opacity_norm * abs(change)
            opacity_c.append(
                opacity if opacity > 0.05 else 0.05)
            cold += 1

    opacities = []
    colors = []
    # hot:
    for i in range(len(x_h)):
        opacities.append(opacity_h[i])
        colors.append('#4c823d')
    # cold:
    for i in range(len(x_c)):
        opacities.append(opacity_c[i])
        colors.append('#d6400a')

    source = ColumnDataSource(data=dict(
        x=x_h + x_c,
        y=y_h + y_c,
        topic=hover_data_h + hover_data_c,
        opacities=opacities,
        colors=colors
    ))
    p = figure(title=header, plot_width=600, plot_height=400, tooltips='@topic')
    p.circle(x='x', y='y', size=13, color='colors',
             alpha='opacities', source=source)
    return p


def create_top_before_graph(top_n):
    global ref_data_name
    header = gd['reports'][0]
    if ref_data_name == "":
        ref_data_name = header.split('(')[1][:-1]
    phrases = gd.values[:, 1]
    importance = gd.values[:, 2]
    valid_phrases, valid_imp = [], []
    ctr = 0
    for i in range(len(phrases)):
        if ctr == top_n:
            break
        p = phrases[i]
        if p in white_list:
            p = cut_phrase(p)
            valid_phrases.append(p)
            valid_imp.append(importance[i])
            ctr += 1

    source = ColumnDataSource(dict(y=valid_phrases[::-1], x=valid_imp[::-1]))
    x_label = 'Importance'
    plot = figure(plot_width=600, plot_height=400, tools="save",
                  x_axis_label=x_label,
                  title=header,
                  y_range=source.data["y"],
                  tooltips='@y<br>@x')
    plot.hbar(source=source, right='x', y='y', left=0, height=0.5,
              color="#335fa5")
    return plot


def create_top_after_graph(topN):
    global target_data_name
    header = gd['reports'][1]
    if target_data_name == "":
        target_data_name = header.split('(')[1][:-1]
    phrases = gd.values[:, 5]
    importance = gd.values[:, 6]
    valid_phrases, valid_imp = [], []
    ctr = 0
    for i in range(len(phrases)):
        if ctr == topN:
            break
        p = phrases[i]
        if p in white_list:
            p = cut_phrase(p)
            valid_phrases.append(p)
            valid_imp.append(importance[i])
            ctr += 1

    source = ColumnDataSource(
        dict(y=valid_phrases[::-1], x=valid_imp[::-1]))
    x_label = 'Importance'
    plot = figure(plot_width=600, plot_height=400, tools="save",
                  x_axis_label=x_label,
                  title=header,
                  y_range=source.data["y"],
                  tooltips='@y<br>@x')
    plot.hbar(source=source, right='x', y='y', left=0, height=0.5,
              color="#335fa5")
    return plot


def handle_selected_graph(selected):
    logger.info("handle selected graph. selected = %s", str(selected))
    label.text = "Please Wait..."
    global apply_filter
    if top_before is not None:
        if apply_filter:
            regenerate_graphs(int(top_n_dropdown.value), int(top_n_clusters_dropdown.value))
            apply_filter = False
        if selected == 0:
            if trend_clustering is None:
                graphs_area.children = [Row(top_before, top_after),
                                        Row(hot_trends, cold_trends),
                                        Row(custom_trends)
                                        ]
            else:
                graphs_area.children = [Row(top_before, top_after),
                                        Row(hot_trends, cold_trends),
                                        Row(trend_clustering, custom_trends),
                                        Row(topic_clustering_before, topic_clustering_after)
                                        ]
        if selected == 1:
            graphs_area.children = [Row(top_before, top_after)]
        if selected == 2:
            graphs_area.children = [Row(hot_trends, cold_trends)]
        if selected == 3:
            if trend_clustering is None:
                graphs_area.children = [Div(text='no word embedding data')]
            else:
                graphs_area.children = [trend_clustering]
        if selected == 4:
            graphs_area.children = [custom_trends]
        if selected == 5:
            if topic_clustering_before is None:
                graphs_area.children = [Div(text='no word embedding data')]
            else:
                graphs_area.children = [Row(topic_clustering_before,
                                        topic_clustering_after)]
        if selected == 6:
            filter_topics_table_source.data = {'topics': filter_rows}
            filter_topics_table_source.selected.indices = get_valid_indices()
            filter_custom_table_source.data = {'topics': filter_rows}
            filter_custom_table_source.selected.indices = get_custom_indices()
            refresh_filter_area()
            apply_filter = False
    label.text = ""


def draw_ui(top_n_phrases, top_n_clusters, active_area):
    logger.info("draw ui. top_n_phrases=%s, top_n_clusters=%s",
                str(top_n_phrases), str(top_n_clusters))
    create_graphs(top_n_phrases, top_n_clusters)
    handle_selected_graph(active_area)


# define callbacks
# pylint: disable=unused-argument
def top_n_phrases_changed_callback(value, old, new):
    logger.info("top n changed to: %s", str(new))
    top_n_dropdown.label = "Show Top " + str(new) + " Phrases"
    draw_ui(new, top_n_clusters_dropdown.value, radio_group_area.active)
    # create_graphs(new)
    # handle_selected_graph(radio_group_area.active)


# pylint: disable=unused-argument
def top_n_clusters_changed_callback(value, old, new):
    global dont_regenerate
    logger.info("top n clusters changed to: %s", str(new))
    top_n_clusters_dropdown.label = "Show Top " + str(new) + " Clusters"
    if not dont_regenerate:
        draw_ui(top_n_dropdown.value, new, radio_group_area.active)
    else:
        logger.info("skip draw_ui")
        dont_regenerate = False


# pylint: disable=unused-argument
def selected_graph_changed_callback(active, old, new):
    global filter_item
    logger.info("selected graph changed to: %s", str(new))
    reset_filters()
    handle_selected_graph(new)


# pylint: disable=unused-argument
def filter_topic_selected_callback(indices, old, new):
    # update all_topics according to new selected items
    logger.info("filter topic callback")
    filter_label.text = "Please Wait..."
    global all_topics, apply_filter
    if filter_item != -1:  # on filter/search state there are None values so need
        # to separate this case
        apply_filter = True
        if filter_item in new:
            all_topics[filter_item][filter_tabs.active + 1] = '1'
        else:
            all_topics[filter_item][filter_tabs.active + 1] = '0'
    elif new != [-1]:
        apply_filter = True
        selected_topics = [filter_topics_table_source.data['topics'][x] for x in new]
        for i, line in enumerate(all_topics):
            if line[0] in selected_topics:
                all_topics[i][1] = '1'
            else:
                all_topics[i][1] = '0'
    filter_label.text = ""


# pylint: disable=unused-argument
def filter_custom_selected_callback(indices, old, new):
    logger.info("filter custom callback")
    filter_label.text = "Please Wait..."
    global all_topics, apply_filter
    if new != [-1]:
        apply_filter = True
        selected_topics = [
            filter_custom_table_source.data['topics'][x] for x in
            new]
        for i, line in enumerate(all_topics):
            if line[0] in selected_topics:
                all_topics[i][2] = '1'
            else:
                all_topics[i][2] = '0'
    filter_label.text = ""


def search_topic_callback():
    global filter_item
    logger.info("search topic callback")
    filter_label.text = ""
    text_to_search = search_input_box.value
    if len(text_to_search) > 0:
        x = search(text_to_search)
        logger.info("x=%s", str(x))
        if x != -1:
            reset_filters()
            filter_item = x
            if filter_tabs.active == 0:
                logger.info("is index %s a valid topic: %s", str(x), str(is_valid_topic(x)))
                filter_topics_table.view.filters[0].indices = [x]
                filter_topics_table_source.data = {'topics': filter_rows}  # just for
                # refreshing the ui
            else:
                filter_custom_table.view.filters[0].indices = [x]
                filter_custom_table_source.data = {
                    'topics': filter_rows}
        else:
            filter_label.text = "no results"
            logger.info("no results")
    refresh_filter_area()


def tab_changed_callback(_):
    reset_filters()


def clear_search_callback():
    reset_filters()
    search_input_box.value = ""
    refresh_filter_area()


def re_analyze_callback():
    re_analyze()


# set callbacks

top_n_dropdown.on_change('value', top_n_phrases_changed_callback)
top_n_clusters_dropdown.on_change('value', top_n_clusters_changed_callback)
radio_group_area.on_change('active', selected_graph_changed_callback)
# pylint: disable=no-member
filter_topics_table_source.selected.on_change('indices', filter_topic_selected_callback)
filter_custom_table_source.selected.on_change('indices', filter_custom_selected_callback)
search_button.on_click(search_topic_callback)
clear_button.on_click(clear_search_callback)
filter_tabs.on_change('active', tab_changed_callback)
analyse_button.on_click(re_analyze_callback)


# start app

draw_ui(top_n_dropdown.value, top_n_clusters_dropdown.value, radio_group_area.active)

doc = curdoc()
main_title = "Trend Analysis"
doc.title = main_title
doc.add_root(grid)
