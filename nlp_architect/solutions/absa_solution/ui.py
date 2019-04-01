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
import numpy as np
import pandas as pd

from bokeh.core.properties import value
from bokeh.document import Document
from bokeh.transform import dodge
from bokeh.layouts import widgetbox, row
from bokeh import layouts
from bokeh.plotting import figure, Figure
from bokeh.models.widgets import DataTable, TableColumn, HTMLTemplateFormatter, RadioButtonGroup
from bokeh.models import HoverTool, Title, Div
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.server.server import Server

from nlp_architect.models.absa.inference.data_types import SentimentDoc, SentimentSentence

POLARITIES = ('POS', 'NEG')


def serve_ui(stats: pd.DataFrame, aspects: pd.Series) -> None:
    """Main function for serving UI application.

    Args:
        stats (pd.DataFrame): Table containing aggregated stats for aspect-polarity pairs
        aspects (pd.Series): List of all aspects
    """
    def _doc_modifier(doc: Document) -> None:
        plot, source = _create_plot(stats, aspects)
        events_table = _create_events_table()
        events_type = RadioButtonGroup(labels=['All Events', 'In-Domain Events'], active=0)

        # pylint: disable=unused-argument
        def _events_handler(attr, old, new):
            _update_events(stats, aspects, events_table, source, events_type.active)

        events_type.on_change('active', _events_handler)
        source.selected.on_change('indices', _events_handler)  # pylint: disable=no-member
        doc.add_root(column(_create_header(), plot, events_type, events_table,
                            sizing_mode="scale_width"))

    print('Opening Bokeh application on http://localhost:5006/')
    server = Server({'/': _doc_modifier})
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def _create_events_table() -> DataTable:
    """Utility function for creating and styling the events table."""
    formatter = HTMLTemplateFormatter(template='''
    <style>
        .AS_POS {color: #0000FF; font-weight: bold;}
        .AS_NEG {color: #0000FF; font-weight: bold;}
        .OP_POS {color: #1aaa0d; font-style: bold;}
        .OP_NEG {color: #f40000;font-style: bold;}
        .NEG_POS {font-style: italic;}
        .NEG_NEG {color: #f40000; font-style: italic;}
        .INT_POS {color: #1aaa0d; font-style: italic;}
        .INT_NEG {color: #f40000; font-style: italic;}
        * {font-size: 105%;}
    </style>
    <%= value %>''')
    columns = [TableColumn(field='POS_events', title='Positive Events', formatter=formatter),
               TableColumn(field='NEG_events', title='Negative Events', formatter=formatter)]
    return DataTable(source=ColumnDataSource(), columns=columns, height=400, index_position=None,
                     width=1500, sortable=False, editable=False, reorderable=False)


def _create_plot(stats: pd.DataFrame, aspects: pd.Series) -> (Figure, ColumnDataSource):
    """Utility function for creating and styling the bar plot."""
    pos_counts, neg_counts = \
        ([stats.loc[(asp, pol, False), 'Quantity'] for asp in aspects] for pol in POLARITIES)
    np.seterr(divide='ignore')
    source = ColumnDataSource(data={'aspects': aspects, 'POS': pos_counts, 'NEG': neg_counts,
                                    'log-POS': np.log2(pos_counts),
                                    'log-NEG': np.log2(neg_counts)})
    np.seterr(divide='warn')
    p = figure(plot_height=150, sizing_mode="scale_width",
               x_range=aspects, toolbar_location='left', tools='save,tap,pan,zoom_in,zoom_out')
    rs = [p.vbar(x=dodge('aspects', -0.207, range=p.x_range), top='log-POS', width=0.4,
                 source=source, color="limegreen", legend=value('POS'), name='POS'),
          p.vbar(x=dodge('aspects', 0.207, range=p.x_range), top='log-NEG', width=0.4,
                 source=source, color="orangered", legend=value('NEG'), name='NEG')]
    for r in rs:
        p.add_tools(HoverTool(tooltips=[('Aspect', '@aspects'), (r.name, '@' + r.name)],
                              renderers=[r]))
    p.add_layout(Title(text=' ' * 5 + 'Log Scale', align='left', text_font_size='20px'), 'left')
    p.yaxis.ticker = []
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    return p, source


def _update_events(stats: pd.DataFrame, aspects: pd.Series, events: DataTable,
                   source: ColumnDataSource, in_domain: bool) -> None:
    """Utility function for updating the content of the events table."""
    i = source.selected.indices
    events.source.data.update({pol + '_events': stats.loc[aspects[i[0]], pol, in_domain]
                               ['Sent_1':].replace(np.nan, '') if i else [] for pol in POLARITIES})


def _create_header() -> layouts.Row:
    """Utility function for creating and styling the header row in the UI layout."""
    lib_github = Div(text="<a href='https://github.com/NervanaSystems/nlp-architect' "
                     "style='text-decoration:none'>NLP ARCHITECT by IntelAI</a>",
                     style={'font-size': '140%', 'color': 'darkblue', 'font-weight': 'bold'})
    title = Div(text="Aspect-Based Sentiment Analysis",
                style={'font-size': '200%', 'color': 'royalblue', 'font-weight': 'bold'})
    return row(widgetbox(lib_github, width=600), widgetbox(title, width=700))


def _ui_format(sent: SentimentSentence, doc: SentimentDoc) -> str:
    """Get sentence as HTML with 4 classes: aspects, opinions, negations and intensifiers."""
    text = doc.doc_text[sent.start: sent.end + 1]
    seen = set()
    for term in sorted([t for e in sent.events for t in e], key=lambda t: t.start)[::-1]:
        if term.start not in seen:
            seen.add(term.start)
            start = term.start - sent.start
            end = start + term.len
            label = term.type.value + '_' + term.polarity.value
            text = ''.join((text[:start], '<span class="', label, '">', text[start: end],
                            '</span>', text[end:]))
    return text
