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
import socket
import pickle
import time
import json

import pandas
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, Row
from bokeh.models.widgets import Button, DataTable, TableColumn, CheckboxGroup, MultiSelect
from bokeh.models.widgets.inputs import TextInput
from bokeh.io import curdoc

import solutions.set_expansion.ui.settings as settings

vocab = None
np2id, id2group, id2rep = {}, {}, {}
vocab_dict = {}
cut_vocab_dict = {}
max_visible_phrases = 5000
working_text = 'please wait...'
fetching_text = 'fetching vocabulary from server. please wait...'
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


# create ui components

seed_input_title = 'Please enter a comma separated seed list of terms:'
seed_input_box = TextInput(
    title=seed_input_title, value="USA, Israel, France", width=450, css_classes=["seed-input"])
search_input_box = TextInput(title="Search:", value="", width=300)
expand_button = Button(label="Expand", button_type="success", width=150)
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
    labels=["Show extracted phrases"], active=[], width=400, css_classes=['checkbox_group'])
search_box_area = column(children=[Div(width=200)])
export_working_label = Div(text="", style={'color': 'red'})
getvocab_working_label = Div(
    text="", style={'color': 'blue', 'padding-top': '0px', 'font-size': '15px'})
search_working_label = Div(
    text="", style={'color': 'blue', 'padding-bottom': '0px', 'font-size': '15px'})
expand_working_label = Div(
    text="", style={
        'color': 'blue', 'padding-top': '7px', 'padding-left': '10px', 'font-size': '15px'})
clear_working_label = Div(
    text="", style={
        'color': 'blue', 'padding-top': '30px', 'padding-left': '20px', 'font-size': '15px'})
seed_check_label = Div(
    text='', style={'font-size': '15px'}, height=20, width=500)
seed_layout = Row(
    seed_input_box, column(Div(height=0, width=0), clear_seed_button), clear_working_label)
table_layout = Row(
    expand_table, Div(width=25), column(Div(height=350), export_button, export_working_label))
table_area = column(children=[table_layout])
phrases_area = column(children=[search_working_label, Div(width=300)])
checkbox_layout = column(checkbox_group, getvocab_working_label)
grid = layout(
    [
        [Div(width=500), Div(text="<H1>Set Expansion Demo</H1>")],
        [checkbox_layout, seed_layout], [search_box_area, Div(width=370),
                                         expand_button, expand_working_label],
        [phrases_area, Div(width=100), table_area]
    ]
)


# define callbacks

def get_vocab():
    """
    Get vocabulary of the np2vec model from the server and load grouping data
    """
    global vocab, np2id, id2group, id2rep
    # print('loading grouping info')
    # with open('np2id') as np2id_file:
    #     np2id = json.load(np2id_file)
    # # with open('id2group') as f:
    # #     id2group = json.load(f)
    # print('np2id loaded. loading id2rep...')
    # with open('id2rep') as id2rep_file:
    #     id2rep = json.load(id2rep_file)
    print('sending get_vocab request to server...')
    received = send_request_to_server('get_vocab')
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
        print('sending request')
        sock.sendall(bytes(request + "\n", "utf-8"))
        # Receive data from the server and shut down
        data = b""
        ctr = 0
        while True:
            packet = sock.recv(134217728)
            print(str(ctr) + '. received: ' + str(len(packet)))
            ctr += 1
            if not packet:
                break
            data += packet
        print('got response, uncompressing')
        received = pickle.loads(data)
        # print("Received: {}".format(received))
        return received
    except EOFError:
        print('No data received')
    finally:
        sock.close()


def row_selected_callback(attr, old, new):
    global clear_flag
    if not clear_flag and expand_table_source.data != empty_table:
        print('row selected callback')
        print('old indices=' + str(old.indices))
        print('new indices=' + str(new.indices))
        global all_selected_phrases
        # sync phrases lists:
        old_phrases = [expand_table_source.data['res'][p] for p in old.indices]
        new_phrases = [expand_table_source.data['res'][p] for p in new.indices]
        print('selected_expand was updated: old=' + str(
            old_phrases) + ' ,new=' + str(new_phrases))
        # phrase was de-selected from expand list:
        for o in old_phrases:
            if o not in new_phrases and \
                    (vocab is not None and vocab_dict[o] in phrases_list.value):
                print('removing ' + o + 'from vocab selected')
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
    print('update selected phrases')
    global all_selected_phrases
    updated_selected_phrases = all_selected_phrases[:]
    selected_expand = [expand_table_source.data['res'][i] for
                       i in expand_table_source.selected.indices if
                       expand_table_source.data['res'][i] != '']
    selected_vocab = phrases_list.value
    print('selected expand=' + str(selected_expand))
    print('selected vocab=' + str(selected_vocab))
    print('current all_selected_phrases=' + str(all_selected_phrases))
    for x in all_selected_phrases:
        print('x=' + x)
        if (x in expand_table_source.data['res'] and x not in selected_expand) or (
            vocab is not None and (vocab_dict[x] in phrases_list.options) and (
                vocab_dict[x] not in selected_vocab
            )
        ):
            print('removing ' + x)
            updated_selected_phrases.remove(x)
    for e in selected_expand:
        if e not in updated_selected_phrases:
            print('adding ' + e)
            updated_selected_phrases.append(e)
    for v in selected_vocab:
        full_v = cut_vocab_dict[v]
        if full_v not in updated_selected_phrases:
            print('adding ' + full_v)
            updated_selected_phrases.append(full_v)
    all_selected_phrases = updated_selected_phrases[:]
    print('all_selected_phrases list was updated: ' + str(all_selected_phrases))


def show_phrases_callback(checked_value):
    global search_box_area, phrases_area
    if len(checked_value) == 1:
        if vocab is None:
            getvocab_working_label.text = fetching_text
            get_vocab()
        if not phrases_list.options:
            getvocab_working_label.text = working_text
            phrases_list.options = list(
                cut_vocab_dict.keys())[0:max_visible_phrases]  # show the cut representation
        search_box_area.children = [search_input_box]
        phrases_area.children = [search_working_label, phrases_list]
        getvocab_working_label.text = ''
    else:
        search_box_area.children = []
        phrases_area.children = []


def get_expand_results_callback():
    """
    Send to the server the seed to expand and set the results in the expand
    table.
    """
    print('### new expand request')
    expand_working_label.text = working_text
    global seed_check_text, table_area
    try:
        if vocab is None:
            expand_working_label.text = fetching_text
            get_vocab()
        seed_check_label.text = ''
        table_area.children = [table_layout]
        seed = seed_input_box.value
        print('input seed: ' + seed)
        if seed == '':
            expand_table_source.data = empty_table
            return
        if vocab is not None:
            seed_words = [x.strip() for x in seed.split(',')]
            bad_words = ''
            for w in seed_words:
                # norm = None
                # if w in np2id.keys():
                #     norm = np2id[w]
                # if norm is None or norm not in id2rep or id2rep[norm] not in vocab:
                #     bad_words += ("'" + w + "',")
                res = send_request_to_server('in_vocab,' + w)
                if res is False:
                    bad_words += ("'" + w + "',")
            if bad_words != '':
                seed_check_label.text = 'the words: <span class="bad-word">' \
                                        + bad_words[:-1] \
                                        + '</span> are not in the vocabulary and will be ignored'
                print('setting table area')
                table_area.children = [seed_check_label, table_layout]
        print('sending expand request to server with seed= ' + seed)
        received = send_request_to_server(seed)
        if received is not None:
            res = [x[0] for x in received]
            scores = [y[1] for y in received]
            print('setting table data')
            expand_table_source.data = {
                'res': res,
                'score': scores
            }
        else:
            print('Nothing received from server')
    except Exception as e:
        print('Exception: ' + str(e))
    finally:
        expand_working_label.text = ''


def search_callback(value, old, new):
    search_working_label.text = working_text
    print('search vocab')
    global vocab, phrases_list, all_selected_phrases, search_flag
    search_flag = True
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
    print('selected vocab after search=' + str(phrases_list.value))
    search_working_label.text = ''
    search_flag = False


def vocab_phrase_selected_callback(attr, old_selected, new_selected):
    print('vocab selected')
    global clear_flag
    if not clear_flag:
        global all_selected_phrases, search_flag
        if(search_flag):
            return
        print('selected_vocab was updated: old=' + str(
            old_selected) + ' ,new=' + str(new_selected))
        # sync expand table:
        # phrase was de-selected from vocab list:
        expand_selected = [expand_table_source.data['res'][p] for
                           p in expand_table_source.selected.indices]
        for o in old_selected:
            full_o = cut_vocab_dict[o]
            if o not in new_selected and full_o in expand_selected:
                print(full_o + ' removed from vocab selected and exists in expand selected')
                print('removing ' + full_o
                      + 'from expand selected indices. index='
                      + str(expand_table_source.data['res'].index(full_o)))
                print('current expand indices: ' + str(expand_table_source.selected.indices))
                expand_table_source.selected.indices.remove(
                    expand_table_source.data['res'].index(full_o))
                print('new expand indices: ' + str(expand_table_source.selected.indices))
                break
        # new phrase was selected from vocab list:
        for n in new_selected:
            full_n = cut_vocab_dict[n]
            print('selected phrase=' + n + ', full phrase=' + full_n)
            if n not in old_selected and full_n in \
                    expand_table_source.data['res'] and full_n not in expand_selected:
                expand_table_source.selected.indices.append(
                    expand_table_source.data['res'].index(full_n))
                break
        update_all_selected_phrases()
        seed_input_box.value = get_selected_phrases_for_seed()


def clear_seed_callback():
    print('clear')
    clear_working_label.text = working_text
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
    clear_working_label.text = ''


def export_data_callback():
    if expand_table_source.data == empty_table:
        export_working_label.text = 'Nothing to export'
        time.sleep(1)
        export_working_label.text = ''
    elif export_working_label.text != working_text:
        path = settings.export_path
        print('saving expansion results to: ' + path)
        export_working_label.style = {'color': 'red'}
        export_working_label.text = working_text
        table_df = pandas.DataFrame(expand_table_source.data)
        table_df.to_csv(path)
        export_working_label.style = {'color': 'green'}
        export_working_label.text = 'Done!'
        time.sleep(1)
        export_working_label.text = ''


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


def expand_data_changed_callback(attr, old, new):
    """
    remove the selected indices when table is empty
    """
    if old == empty_table:
        expand_table_source.selected.indices = []


# set callbacks

expand_button.on_click(get_expand_results_callback)
expand_table_source.on_change('selected', row_selected_callback)
expand_table_source.on_change('data', expand_data_changed_callback)
checkbox_group.on_click(show_phrases_callback)
search_input_box.on_change('value', search_callback)
phrases_list.on_change('value', vocab_phrase_selected_callback)
clear_seed_button.on_click(clear_seed_callback)
export_button.on_click(export_data_callback)
# table_area.on_change('children', table_area_change_callback)


# arrange components in page

doc = curdoc()
main_title = "Set Expansion Demo"
doc.title = main_title
doc.add_root(grid)


# present initial example:
# get_expand_results_callback()
