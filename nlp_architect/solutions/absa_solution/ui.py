# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
import base64
import io
import os
import json
from os.path import dirname, join

import pandas as pd
import numpy as np
from bokeh import layouts
from bokeh.document import Document
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models import Div, CustomJS, HoverTool, Title, ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, TextInput
from bokeh.models.widgets import HTMLTemplateFormatter, RadioButtonGroup
from bokeh.models.widgets import Dropdown
from bokeh.models.widgets import Tabs, Panel
from bokeh.plotting import Figure, figure
from bokeh.server.server import Server
from bokeh.transform import dodge
from bokeh.core.properties import value
from tornado.web import StaticFileHandler

from nlp_architect.models.absa import LEXICONS_OUT
from nlp_architect.models.absa.train.acquire_terms import AcquireTerms
from nlp_architect.models.absa.train.train import TrainSentiment
from nlp_architect.solutions.absa_solution import SENTIMENT_OUT
from nlp_architect.solutions.absa_solution.sentiment_solution import SentimentSolution
from nlp_architect.models.absa.inference.data_types import SentimentDoc, SentimentSentence

POLARITIES = ('POS', 'NEG')


# pylint: disable=global-variable-undefined
def serve_absa_ui() -> None:
    """Main function for serving UI application.
    """

    def _doc_modifier(doc: Document) -> None:
        grid = _create_ui_components()
        doc.add_root(grid)

    print('Opening Bokeh application on http://localhost:5006/')
    server = Server({'/': _doc_modifier}, websocket_max_message_size=5000 * 1024 * 1024,
                    extra_patterns=[('/style/(.*)', StaticFileHandler,
                                     {'path': os.path.normpath(os.path.dirname(__file__)
                                                               + '/style')})])
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def _create_header(train_dropdown,
                   inference_dropdown, text_status) -> layouts.Row:
    """Utility function for creating and styling the header row in the UI layout."""

    architect_logo = Div(text='<a href="http://nlp_architect.nervanasys.com"> <img border="0" '
                              'src="style/nlp_architect.jpg" width="200"></a> by Intel® AI Lab',
                         style={'margin-left': '500px', 'margin-top': '20px', 'font-size': '110%',
                                'text-align': 'center'})
    css_link = Div(text="<link rel='stylesheet' type='text/css' href='style/lexicon_manager.css'>",
                   style={'font-size': '0%'})

    js_script = Div(text="<input type='file' id='inputOS' hidden='true'>")

    title = Div(text="ABSApp",
                style={'font-size': '300%', 'color': 'royalblue', 'font-weight': 'bold',
                       'margin-left': '500px'})

    return row(column(row(children=[train_dropdown, lexicons_dropdown, inference_dropdown],
                          width=500), row(text_status)), css_link, js_script,
               widgetbox(title, width=900, height=84),
               widgetbox(architect_logo, width=400, height=84))


def empty_table(*headers):
    return ColumnDataSource(data={header: 19 * [''] for header in headers})


def _create_ui_components() -> (Figure, ColumnDataSource):  # pylint: disable=too-many-statements
    global asp_table_source, asp_filter_src, op_table_source, op_filter_src
    global stats, aspects, tabs, lexicons_dropdown
    stats = pd.DataFrame(columns=['Quantity', 'Score'])
    aspects = pd.Series([])

    def new_col_data_src():
        return ColumnDataSource({'file_contents': [], 'file_name': []})

    large_text = HTMLTemplateFormatter(template='''<div><%= value %></div>''')

    def data_column(title):
        return TableColumn(field=title, title='<span class="header">'
                           + title + '</span>', formatter=large_text)

    asp_table_columns = [data_column('Term'), data_column(
        'Alias1'), data_column('Alias2'), data_column('Alias3')]
    op_table_columns = [data_column('Term'), data_column('Score'), data_column('Polarity')]

    asp_table_source = empty_table('Term', 'Alias1', 'Alias2', 'Alias3')
    asp_filter_src = empty_table('Term', 'Alias1', 'Alias2', 'Alias3')
    asp_src = new_col_data_src()

    op_table_source = empty_table('Term', 'Score', 'Polarity', 'Polarity')
    op_filter_src = empty_table('Term', 'Score', 'Polarity', 'Polarity')
    op_src = new_col_data_src()

    asp_table = DataTable(
        source=asp_table_source, selectable='checkbox', columns=asp_table_columns,
        editable=True, width=600, height=500)
    op_table = DataTable(
        source=op_table_source, selectable='checkbox', columns=op_table_columns,
        editable=True, width=600, height=500)

    asp_examples_box = _create_examples_table()
    op_examples_box = _create_examples_table()
    asp_layout = layout([[asp_table, asp_examples_box]])
    op_layout = layout([[op_table, op_examples_box]])
    asp_tab = Panel(child=asp_layout, title="Aspect Lexicon")
    op_tab = Panel(child=op_layout, title="Opinion Lexicon")
    tabs = Tabs(tabs=[asp_tab, op_tab], width=700, css_classes=['mytab'])

    lexicons_menu = [("Open", "open"), ("Save", "save")]
    lexicons_dropdown = Dropdown(label="Edit Lexicons", button_type="success", menu=lexicons_menu,
                                 width=140, height=31, css_classes=['mybutton'])

    train_menu = [("Parsed Data", "parsed"), ("Raw Data", "raw")]
    train_dropdown = Dropdown(label="Extract Lexicons", button_type="success", menu=train_menu,
                              width=162, height=31, css_classes=['mybutton'])

    inference_menu = [("Parsed Data", "parsed"), ("Raw Data", "raw")]
    inference_dropdown = Dropdown(label="Classify", button_type="success", menu=inference_menu,
                                  width=140, height=31, css_classes=['mybutton'])

    text_status = TextInput(value="Select training data", title="Train Run Status:",
                            css_classes=['statusText'])
    text_status.visible = False

    train_src = new_col_data_src()
    infer_src = new_col_data_src()

    with open(join(dirname(__file__), "dropdown.js")) as f:
        args = dict(clicked=lexicons_dropdown, asp_filter=asp_filter_src, op_filter=op_filter_src,
                    asp_src=asp_src, op_src=op_src, tabs=tabs,
                    text_status=text_status,
                    train_src=train_src, infer_src=infer_src, train_clicked=train_dropdown,
                    infer_clicked=inference_dropdown, opinion_lex_generic="")
        code = f.read()

    args['train_clicked'] = train_dropdown
    train_dropdown.js_on_change('value', CustomJS(args=args, code=code))

    args['train_clicked'] = inference_dropdown
    inference_dropdown.js_on_change('value', CustomJS(args=args, code=code))

    args['clicked'] = lexicons_dropdown
    lexicons_dropdown.js_on_change('value', CustomJS(args=args, code=code))

    def update_filter_source(table_source, filter_source):
        df = table_source.to_df()
        sel_inx = sorted(table_source.selected.indices)
        df = df.iloc[sel_inx, 1:]
        new_source = ColumnDataSource(df)
        filter_source.data = new_source.data

    def update_examples_box(data, examples_box, old, new):
        examples_box.source.data = {'Examples': []}
        unselected = list(set(old) - set(new))
        selected = list(set(new) - set(old))
        if len(selected) <= 1 and len(unselected) <= 1:
            examples_box.source.data.update(
                {'Examples': [str(data.iloc[unselected[0], i]) for i in range(4, 24)]
                 if len(unselected) != 0 else [str(data.iloc[selected[0], i]) for i
                                               in range(4, 24)]})

    def asp_selected_change(_, old, new):
        global asp_filter_src, asp_table_source, aspects_data
        update_filter_source(asp_table_source, asp_filter_src)
        update_examples_box(aspects_data, asp_examples_box, old, new)

    def op_selected_change(_, old, new):
        global op_filter_src, op_table_source, opinions_data
        update_filter_source(op_table_source, op_filter_src)
        update_examples_box(opinions_data, op_examples_box, old, new)

    def read_csv(file_src, headers=False, index_cols=False, readCSV=True):
        if readCSV:
            raw_contents = file_src.data['file_contents'][0]

            if len(raw_contents.split(",")) == 1:
                b64_contents = raw_contents
            else:
                # remove the prefix that JS adds
                b64_contents = raw_contents.split(",", 1)[1]
            file_contents = base64.b64decode(b64_contents)
            return pd.read_csv(io.BytesIO(file_contents), encoding="ISO-8859-1",
                               keep_default_na=False, na_values={None},
                               engine='python', index_col=index_cols,
                               header=0 if headers else None)
        return file_src

    def read_parsed_files(file_content, file_name):
        try:
            # remove the prefix that JS adds
            b64_contents = file_content.split(",", 1)[1]
            file_content = base64.b64decode(b64_contents)
            with open(SENTIMENT_OUT / file_name, 'w') as json_file:
                data_dict = json.loads(file_content.decode("utf-8"))
                json.dump(data_dict, json_file)
        except Exception as e:
            print(str(e))

    # pylint: disable=unused-argument
    def train_file_callback(attr, old, new):
        global train_data
        SENTIMENT_OUT.mkdir(parents=True, exist_ok=True)
        train = TrainSentiment(parse=True, rerank_model=None)
        if len(train_src.data['file_contents']) == 1:
            train_data = read_csv(train_src, index_cols=0)
            file_name = train_src.data['file_name'][0]
            raw_data_path = SENTIMENT_OUT / file_name
            train_data.to_csv(raw_data_path, header=False)
            print(f'Running_SentimentTraining on data...')
            train.run(data=raw_data_path)
        else:
            f_contents = train_src.data['file_contents']
            f_names = train_src.data['file_name']
            raw_data_path = SENTIMENT_OUT / train_src.data['file_name'][0].split('/')[0]
            if not os.path.exists(raw_data_path):
                os.makedirs(raw_data_path)
            for f_content, f_name in zip(f_contents, f_names):
                read_parsed_files(f_content, f_name)
            print(f'Running_SentimentTraining on data...')
            train.run(parsed_data=raw_data_path)

        text_status.value = "Lexicon extraction completed"

        with io.open(AcquireTerms.acquired_aspect_terms_path, "r") as fp:
            aspect_data_csv = fp.read()
        file_data = base64.b64encode(str.encode(aspect_data_csv))
        file_data = file_data.decode("utf-8")
        asp_src.data = {'file_contents': [file_data], 'file_name': ['nameFile.csv']}

        out_path = LEXICONS_OUT / 'generated_opinion_lex_reranked.csv'
        with io.open(out_path, "r") as fp:
            opinion_data_csv = fp.read()
        file_data = base64.b64encode(str.encode(opinion_data_csv))
        file_data = file_data.decode("utf-8")
        op_src.data = {'file_contents': [file_data], 'file_name': ['nameFile.csv']}

    def show_analysis() -> None:
        global stats, aspects, plot, source, tabs
        plot, source = _create_plot()
        events_table = _create_events_table()

        # pylint: disable=unused-argument
        def _events_handler(attr, old, new):
            _update_events(events_table, events_type.active)

        # Toggle display of in-domain / All aspect mentions
        events_type = RadioButtonGroup(labels=['All Events', 'In-Domain Events'], active=0)

        analysis_layout = layout([[plot], [events_table]])

        # events_type display toggle disabled
        # analysis_layout = layout([[plot],[events_type],[events_table]])

        analysis_tab = Panel(child=analysis_layout, title="Analysis")
        tabs.tabs.insert(2, analysis_tab)
        tabs.active = 2
        events_type.on_change('active', _events_handler)
        source.selected.on_change('indices', _events_handler)  # pylint: disable=no-member

    # pylint: disable=unused-argument
    def infer_file_callback(attr, old, new):

        # run inference on input data and current aspect/opinion lexicons in view
        global infer_data, stats, aspects

        SENTIMENT_OUT.mkdir(parents=True, exist_ok=True)

        df_aspect = pd.DataFrame.from_dict(asp_filter_src.data)
        aspect_col_list = ['Term', 'Alias1', 'Alias2', 'Alias3']
        df_aspect = df_aspect[aspect_col_list]
        df_aspect.to_csv(SENTIMENT_OUT / 'aspects.csv', index=False, na_rep="NaN")

        df_opinion = pd.DataFrame.from_dict(op_filter_src.data)
        opinion_col_list = ['Term', 'Score', 'Polarity', 'isAcquired']
        df_opinion = df_opinion[opinion_col_list]
        df_opinion.to_csv(SENTIMENT_OUT / 'opinions.csv', index=False, na_rep="NaN")

        solution = SentimentSolution()

        if len(infer_src.data['file_contents']) == 1:
            infer_data = read_csv(infer_src, index_cols=0)
            file_name = infer_src.data['file_name'][0]
            raw_data_path = SENTIMENT_OUT / file_name
            infer_data.to_csv(raw_data_path, header=False)
            print(f'Running_SentimentInference on data...')
            text_status.value = "Running classification on data..."
            stats = solution.run(data=raw_data_path,
                                 aspect_lex=SENTIMENT_OUT / 'aspects.csv',
                                 opinion_lex=SENTIMENT_OUT / 'opinions.csv')
        else:
            f_contents = infer_src.data['file_contents']
            f_names = infer_src.data['file_name']
            raw_data_path = SENTIMENT_OUT / infer_src.data['file_name'][0].split('/')[0]
            if not os.path.exists(raw_data_path):
                os.makedirs(raw_data_path)
            for f_content, f_name in zip(f_contents, f_names):
                read_parsed_files(f_content, f_name)
            print(f'Running_SentimentInference on data...')
            text_status.value = "Running classification on data..."
            stats = solution.run(parsed_data=raw_data_path,
                                 aspect_lex=SENTIMENT_OUT / 'aspects.csv',
                                 opinion_lex=SENTIMENT_OUT / 'opinions.csv')

        aspects = pd.read_csv(SENTIMENT_OUT / 'aspects.csv', encoding='utf-8')['Term']
        text_status.value = "Classification completed"
        show_analysis()

    # pylint: disable=unused-argument
    def asp_file_callback(attr, old, new):
        global aspects_data, asp_table_source
        aspects_data = read_csv(asp_src, headers=True)
        # Replaces None values by empty string
        aspects_data = aspects_data.fillna('')
        new_source = ColumnDataSource(aspects_data)
        asp_table_source.data = new_source.data
        asp_table_source.selected.indices = list(range(len(aspects_data)))

    # pylint: disable=unused-argument
    def op_file_callback(attr, old, new):
        global opinions_data, op_table_source, lexicons_dropdown, df_opinion_generic
        df = read_csv(op_src, headers=True)
        # Replaces None values by empty string
        df = df.fillna('')
        # Placeholder for generic opinion lexicons from the given csv file
        df_opinion_generic = df[df['isAcquired'] == 'N']
        # Update the argument value for the callback customJS
        lexicons_dropdown.js_property_callbacks.get(
            'change:value')[0].args['opinion_lex_generic'] \
            = df_opinion_generic.to_dict(orient='list')
        opinions_data = df[df['isAcquired'] == 'Y']
        new_source = ColumnDataSource(opinions_data)
        op_table_source.data = new_source.data
        op_table_source.selected.indices = list(range(len(opinions_data)))

    # pylint: disable=unused-argument
    def txt_status_callback(attr, old, new):
        print("Previous label: " + old)
        print("Updated label: " + new)

    text_status.on_change("value", txt_status_callback)

    asp_src.on_change('data', asp_file_callback)
    # pylint: disable=no-member
    asp_table_source.selected.on_change(
        'indices', asp_selected_change)

    op_src.on_change('data', op_file_callback)
    op_table_source.selected.on_change('indices', op_selected_change)  # pylint: disable=no-member

    train_src.on_change('data', train_file_callback)
    infer_src.on_change('data', infer_file_callback)

    return layout([
        [_create_header(train_dropdown, inference_dropdown, text_status)],
        [tabs]])


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
    </style>
    <%= value %>''')
    columns = [TableColumn(field='POS_events', title='Positive Examples', formatter=formatter),
               TableColumn(field='NEG_events', title='Negative Examples', formatter=formatter)]
    return DataTable(source=ColumnDataSource(), columns=columns, height=400, index_position=None,
                     width=2110, sortable=False, editable=True, reorderable=False)


def _create_plot() -> (Figure, ColumnDataSource):
    """Utility function for creating and styling the bar plot."""
    global source, aspects, stats
    pos_counts, neg_counts = \
        ([stats.loc[(asp, pol, False), 'Quantity'] for asp in aspects] for pol in POLARITIES)
    np.seterr(divide='ignore')
    source = ColumnDataSource(data={'aspects': aspects, 'POS': pos_counts, 'NEG': neg_counts,
                                    'log-POS': np.log2(pos_counts),
                                    'log-NEG': np.log2(neg_counts)})
    np.seterr(divide='warn')
    p = figure(plot_height=145, sizing_mode="scale_width",
               x_range=aspects, toolbar_location='right', tools='save, tap')
    rs = [p.vbar(x=dodge('aspects', -0.207, range=p.x_range), top='log-POS', width=0.4,
                 source=source, color="limegreen", legend=value('POS'), name='POS'),
          p.vbar(x=dodge('aspects', 0.207, range=p.x_range), top='log-NEG', width=0.4,
                 source=source, color="orangered", legend=value('NEG'), name='NEG')]
    for r in rs:
        p.add_tools(HoverTool(tooltips=[('Aspect', '@aspects'), (r.name, '@' + r.name)],
                              renderers=[r]))
    p.add_layout(Title(text=' ' * 7 + 'Sentiment Count (log scale)', align='left',
                       text_font_size='23px'), 'left')
    p.yaxis.ticker = []
    p.y_range.start = 0
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_text_font_size = "20pt"
    p.legend.label_text_font_size = '20pt'
    return p, source


def _update_events(events: DataTable, in_domain: bool) -> None:
    """Utility function for updating the content of the events table."""
    i = source.selected.indices
    events.source.data.update({pol + '_events': stats.loc[aspects[i[0]], pol, in_domain]
                               ['Sent_1':].replace(np.nan, '') if i else [] for pol in POLARITIES})


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


def _create_examples_table() -> DataTable:
    """Utility function for creating and styling the events table."""

    formatter = HTMLTemplateFormatter(template='''
    <style>
        .AS {color: #0000FF; font-weight: bold;}
        .OP {color: #0000FF; font-weight: bold;}
    </style>
    <div><%= value %></div>''')
    columns = [TableColumn(field='Examples', title='<span class="header">Examples</span>',
                           formatter=formatter)]
    empty_source = ColumnDataSource()
    empty_source.data = {'Examples': []}
    return DataTable(source=empty_source, columns=columns, height=500, index_position=None,
                     width=1500, sortable=False, editable=False, reorderable=False,
                     header_row=True)


if __name__ == '__main__':
    serve_absa_ui()
