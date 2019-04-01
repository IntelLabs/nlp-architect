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

import socket
import pickle
import logging
import sys
import re
from os.path import dirname, join

from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, Row, CustomJS
from bokeh.models.widgets import Button, DataTable, TableColumn, CheckboxGroup, MultiSelect
from bokeh.models.widgets.inputs import TextInput
from bokeh.io import curdoc

import nlp_architect.solutions.set_expansion.ui.settings as settings

# pylint: skip-file
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

vocab = None
vocab_dict = {}
cut_vocab_dict = {}
max_visible_phrases = 5000
working_text = 'please wait...'
fetching_text = 'Fetching vocabulary from server (one time only), this can take few minutes...'
seed_check_text = ''
all_selected_phrases = []
search_flag = False
max_phrase_length = 40
clear_flag = False
expand_columns = [
    TableColumn(field="res", title="Results"),
    TableColumn(field="score", title="Score")
]
empty_table = {'res': 15 * [''], 'score': 15 * ['']}
checkbox_label = "Show extracted term groups" if settings.grouping else "Show extracted phrases"

# create ui components

seed_input_title = 'Please enter a comma separated seed list of terms:'
seed_input_box = TextInput(
    title=seed_input_title, value="", width=450, css_classes=["seed-input"])
annotation_input = TextInput(title="Please enter text to annotate:", value="", width=400,
                             height=80, css_classes=["annotation-input"])
annotation_output = Div(text='', height=30, width=500, style={'padding-left': '35px'})
annotate_button = Button(label="Annotate", button_type="success", width=150,
                         css_classes=["annotation-button"])
group_info_box = Div(text='', height=30, css_classes=["group-div"])
search_input_box = TextInput(title="Search:", value="", width=300)
expand_button = Button(label="Expand", button_type="success", width=150,
                       css_classes=["expand-button"])
clear_seed_button = Button(
    label="Clear", button_type="success", css_classes=['clear_button'], width=50)
export_button = Button(
    label="Export", button_type="success", css_classes=['export_button'], width=100)
expand_table_source = ColumnDataSource(data=empty_table)
expand_table = DataTable(
    source=expand_table_source, columns=expand_columns, width=500, css_classes=['expand_table'])
phrases_list = MultiSelect(
    title="", value=[], options=[], width=300, size=27, css_classes=['phrases_list'])
checkbox_group = CheckboxGroup(
    labels=["Text annotation", checkbox_label], active=[], width=400,
    css_classes=['checkbox_group'])
annotate_checkbox = CheckboxGroup(
    labels=["Text annotation"], active=[], width=400, css_classes=['annotate_checkbox'])
search_box_area = column(children=[Div(height=10, width=200)])
working_label = Div(
    text="", style={'color': 'blue', 'font-size': '15px'})
search_working_label = Div(
    text="", style={'color': 'blue', 'padding-bottom': '0px', 'font-size': '15px'})
seed_check_label = Div(
    text='', style={'font-size': '15px'}, height=20, width=500)
table_layout = Row(
    expand_table)
table_area = column(children=[table_layout])
seed_layout = column(Row(seed_input_box, column(Div(height=14, width=0), clear_seed_button)),
                     expand_button, table_area)
annotation_layout = column(children=[])

phrases_area = column(children=[search_working_label, Div(width=300)])
checkbox_layout = column(children=[checkbox_group, phrases_area])
grid = layout(
    [
        [working_label, Div(width=250), Div(text="<h1>Set Expansion Demo</h1>")],
        [checkbox_layout, seed_layout, Div(width=50),
         column(Div(height=0, width=0), annotation_layout)],
        [group_info_box, Div(width=500), export_button]
    ]
)


# define callbacks

def get_vocab():
    """
    Get vocabulary of the np2vec model from the server
    """
    global vocab
    logger.info('sending get_vocab request to server...')
    received = send_request_to_server(['get_vocab'])
    vocab = received
    for p in vocab:
        if len(p) < max_phrase_length:
            vocab_dict[p] = p
            cut_vocab_dict[p] = p
        else:
            vocab_dict[p] = p[:max_phrase_length - 1] + '...'
            cut_vocab_dict[p[:max_phrase_length - 1] + '...'] = p


def send_request_to_server(request):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Connect to server and send data
        sock.connect((settings.expand_host, settings.expand_port))
        logger.info('sending request')
        req_packet = pickle.dumps(request)
        # sock.sendall(bytes(request + "\n", "utf-8"))
        sock.sendall(req_packet)
        # Receive data from the server and shut down
        data = b""
        ctr = 0
        while True:
            packet = sock.recv(134217728)
            logger.info("%s. received: %s", str(ctr), str(len(packet)))
            ctr += 1
            if not packet:
                break
            data += packet
            logger.info('got response, uncompressing')
        received = pickle.loads(data)
        # logger.info("Received: {}".format(received))
        return received
    except EOFError:
        logger.info('No data received')
    finally:
        sock.close()


def row_selected_callback(indices, old, new):
    logger.info('row selected callback')
    global clear_flag, all_selected_phrases
    if not clear_flag and expand_table_source.data != empty_table:
        logger.info('row selected callback. old indices=%s. new indices=%s',
                    str(old), str(new))
        # sync phrases lists:
        old_phrases = [expand_table_source.data['res'][p] for p in old]
        new_phrases = [expand_table_source.data['res'][p] for p in new]
        logger.info('selected_expand was updated: old=%s ,new=%s', str(
            old_phrases), str(new_phrases))
        # phrase was de-selected from expand list:
        for o in old_phrases:
            if o not in new_phrases and \
                    (vocab is not None and vocab_dict[o] in phrases_list.value):
                logger.info('removing %s from vocab selected', o)
                phrases_list.value.remove(vocab_dict[o])
                break
        # new phrase was selected from expand list:
        for n in new_phrases:
            if n not in old_phrases and \
                    (vocab is not None and vocab_dict[n] in
                     phrases_list.options and vocab_dict[n] not in phrases_list.value):
                phrases_list.value.append(vocab_dict[n])
                break
        update_all_selected_phrases()
        seed_input_box.value = get_selected_phrases_for_seed()


def update_all_selected_phrases():
    """
    Sync selected values from both the expand-table and the vocabulary list
    """
    logger.info('update selected phrases')
    global all_selected_phrases
    updated_selected_phrases = all_selected_phrases[:]
    selected_expand = [expand_table_source.data['res'][i] for
                       i in expand_table_source.selected.indices if
                       expand_table_source.data['res'][i] != '']
    selected_vocab = phrases_list.value
    logger.info('selected expand= %s', str(selected_expand))
    logger.info('selected vocab= %s', str(selected_vocab))
    logger.info('current all_selected_phrases= %s', str(all_selected_phrases))
    for x in all_selected_phrases:
        logger.info('x= %s', x)
        if (x in expand_table_source.data['res'] and x not in selected_expand) or (
                vocab is not None and (vocab_dict[x] in phrases_list.options) and (
                vocab_dict[x] not in selected_vocab)
        ):
            logger.info('removing %s', x)
            updated_selected_phrases.remove(x)
    for e in selected_expand:
        if e not in updated_selected_phrases:
            logger.info('adding %s', e)
            updated_selected_phrases.append(e)
    for v in selected_vocab:
        full_v = cut_vocab_dict[v]
        if full_v not in updated_selected_phrases:
            logger.info('adding %s', full_v)
            updated_selected_phrases.append(full_v)
    all_selected_phrases = updated_selected_phrases[:]
    logger.info('all_selected_phrases list was updated: %s', str(all_selected_phrases))


def checkbox_callback(checked_value):
    global search_box_area, phrases_area
    group_info_box.text = ''
    if 0 in checked_value:
        annotation_layout.children = [annotation_input, annotate_button,
                                      annotation_output]
    else:
        annotation_layout.children = []
        annotation_output.text = ""
    if 1 in checked_value:
        if vocab is None:
            working_label.text = fetching_text
            get_vocab()
        if not phrases_list.options:
            working_label.text = working_text
            phrases_list.options = list(
                cut_vocab_dict.keys())[0:max_visible_phrases]  # show the cut representation
        # search_box_area.children = [search_input_box]
        phrases_area.children = [search_input_box, search_working_label, phrases_list]
        working_label.text = ''
    else:
        # search_box_area.children = []
        phrases_area.children = []
        group_info_box.text = ""


def get_expand_results_callback():
    """
    Send to the server the seed to expand and set the results in the expand
    table.
    """
    logger.info('### new expand request')
    working_label.text = working_text
    global seed_check_text, table_area
    try:
        seed_check_label.text = ''
        table_area.children = [table_layout]
        seed = seed_input_box.value
        logger.info('input seed: %s', seed)
        if seed == '':
            expand_table_source.data = empty_table
            return
        seed_words = [x.strip() for x in seed.split(',')]
        bad_words = ''
        for w in seed_words:
            res = send_request_to_server(['in_vocab', w])
            if res is False:
                bad_words += ("'" + w + "',")
        if bad_words != '':
            seed_check_label.text = 'the words: <span class="bad-word">' \
                                    + bad_words[:-1] \
                                    + '</span> are not in the vocabulary and will be ignored'
            logger.info('setting table area')
            table_area.children = [seed_check_label, table_layout]
        logger.info('sending expand request to server with seed= %s', seed)
        received = send_request_to_server(['expand', seed])
        if received is not None:
            res = [x[0] for x in received]
            scores = ["{0:.5f}".format(y[1]) for y in received]
            logger.info('setting table data')
            expand_table_source.data = {
                'res': res,
                'score': scores
            }
        else:
            logger.info('Nothing received from server')
    except Exception as e:
        logger.info('Exception: %s', str(e))
    finally:
        working_label.text = ''


def search_callback(value, old, new):
    group_info_box.text = ''
    search_working_label.text = working_text
    logger.info('search vocab')
    global vocab, phrases_list, all_selected_phrases, search_flag
    search_flag = True
    phrases_list.value = []
    if new == '':
        new_phrases = list(cut_vocab_dict.keys())
    else:
        new_phrases = []
        for x in vocab:
            if x.lower().startswith(new.lower()) and vocab_dict[x] not in new_phrases:
                new_phrases.append(vocab_dict[x])
    phrases_list.options = new_phrases[0:max_visible_phrases]
    if new != '':
        phrases_list.options.sort()
    phrases_list.value = [
        vocab_dict[x] for x in all_selected_phrases if vocab_dict[x] in phrases_list.options]
    logger.info('selected vocab after search= %s', str(phrases_list.value))
    search_working_label.text = ''
    search_flag = False


def vocab_phrase_selected_callback(attr, old_selected, new_selected):
    logger.info('vocab selected')
    if settings.grouping:
        # show group info
        if len(new_selected) == 1:
            res = send_request_to_server(['get_group', new_selected[0]])
            if res is not None:
                group_info_box.text = str(res)
    global clear_flag
    if not clear_flag:
        global all_selected_phrases, search_flag
        if (search_flag):
            return
        logger.info('selected_vocab was updated: old= %s, new= %s', str(
            old_selected), str(new_selected))
        # sync expand table:
        # phrase was de-selected from vocab list:
        expand_selected = [expand_table_source.data['res'][p] for
                           p in expand_table_source.selected.indices]
        for o in old_selected:
            full_o = cut_vocab_dict[o]
            if o not in new_selected and full_o in expand_selected:
                logger.info('%s removed from vocab selected and exists in expand selected', full_o)
                logger.info('removing %s from expand selected indices. index=%s',
                            full_o, str(expand_table_source.data['res'].index(full_o)))
                logger.info('current expand indices: %s',
                            str(expand_table_source.selected.indices))
                expand_table_source.selected.indices.remove(
                    expand_table_source.data['res'].index(full_o))
                logger.info('new expand indices: %s', str(expand_table_source.selected.indices))
                break
        # new phrase was selected from vocab list:
        for n in new_selected:
            full_n = cut_vocab_dict[n]
            logger.info('selected phrase=' + n + ', full phrase=' + full_n)
            if n not in old_selected and full_n in \
                    expand_table_source.data['res'] and full_n not in expand_selected:
                expand_table_source.selected.indices.append(
                    expand_table_source.data['res'].index(full_n))
                break
        update_all_selected_phrases()
        seed_input_box.value = get_selected_phrases_for_seed()


def clear_seed_callback():
    logger.info('clear')
    global all_selected_phrases, table_area, clear_flag
    # table_area.children = []  # needed for refreshing the selections
    clear_flag = True
    seed_input_box.value = ''
    seed_check_label.text = ''
    expand_table_source.selected.indices = []
    phrases_list.value = []
    all_selected_phrases = []
    table_area.children = [table_layout]
    clear_flag = False


def get_selected_phrases_for_seed():
    """
     create the seed string to send to the server
    """
    global all_selected_phrases
    phrases = ''
    for x in all_selected_phrases:
        phrases += x + ', '
    phrases = phrases[:-2]
    return phrases


def expand_data_changed_callback(data, old, new):
    """
    remove the selected indices when table is empty
    """
    if old == empty_table:
        expand_table_source.selected.indices = []


def annotate_callback():
    try:
        annotation_output.text = working_text
        user_text = annotation_input.value
        # if len(user_text) == 0 :
        #    annotation_output.text = "Please provaide valid text to annotate"
        if len(seed_input_box.value) == 0:
            out_text = "No seed to compare to"
        else:
            out_text = user_text
            seed = [x.strip() for x in seed_input_box.value.split(',')]
            res = send_request_to_server(['annotate', seed, user_text])
            logger.info("res:%s", str(res))
            if len(res) == 0:
                out_text = "No results found"
            for np in res:
                pattern = re.compile(r'\b' + np + r'\b')
                out_text = re.sub(pattern, mark_phrase_tag(np), out_text)
        annotation_output.text = out_text
    except Exception as e:
        annotation_output.text = "An error occured"
        logger.error("Error: %s", e)


def mark_phrase_tag(text):
    return '<phrase>' + text + '</phrase>'


# set callbacks

expand_button.on_click(get_expand_results_callback)
expand_table_source.selected.on_change('indices', row_selected_callback)
expand_table_source.on_change('data', expand_data_changed_callback)
checkbox_group.on_click(checkbox_callback)
search_input_box.on_change('value', search_callback)
phrases_list.on_change('value', vocab_phrase_selected_callback)
clear_seed_button.on_click(clear_seed_callback)
with open(join(dirname(__file__), "download.js")) as f:
    code = f.read()
export_button.callback = CustomJS(args=dict(source=expand_table_source),
                                  code=code)
annotate_button.on_click(annotate_callback)
# table_area.on_change('children', table_area_change_callback)

# arrange components in page

doc = curdoc()
main_title = "Set Expansion Demo"
doc.title = main_title
doc.add_root(grid)
