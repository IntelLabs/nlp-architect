from bokeh.layouts import column, widgetbox, gridplot, layout, Spacer
from bokeh.models import ColumnDataSource, Div, Row
from bokeh.models.widgets import Button, DataTable, TableColumn, RadioGroup, CheckboxGroup, MultiSelect, Toggle,HTMLTemplateFormatter
from bokeh.models.widgets.inputs import TextInput
from bokeh.models.widgets.tables import BooleanFormatter, CheckboxEditor
from bokeh.core.enums import Enumeration, enumeration
from bokeh.core.properties import Enum
from bokeh.events import *
from bokeh.io import curdoc
from bokeh.models.selections import Selection
# from nlp_architect.utils.text_preprocess import simple_normalizer
import numpy as np
import pandas
import socket
import pickle
import csv
import sys
import os
import time
import pprint

expand_host = 'localhost'
port = 1111
out_path = "export.csv"
hash2group = {}
all_phrases = None
all_phrases_dict = {}
all_cut_phrases_dict = {}
max_visible_phrases = 5000
working_text = 'please wait...'
seed_check_text = ''
all_selected_phrases = []
search_flag = False
max_phrase_length = 100
clear_flag = False
expand_columns = [
    TableColumn(field="res", title="Results"),
    TableColumn(field="score", title="Score")
]

# create ui components
seed_input_title = 'Please enter a comma separated seed list of terms:'
seed_input_box = TextInput(title=seed_input_title, value="USA, Israel, France", width=450, css_classes=["seed-input"])
search_input_box = TextInput(title="Search:", value="", width=300)
expand_button = Button(label="Expand", button_type="success", width=150)
clear_seed_button = Button(label="Clear", button_type="success", css_classes=['clear_button'], width=50)
export_button = Button(label="Export", button_type="success", css_classes=['export_button'], width=100)
expand_table_source = ColumnDataSource(data={'res':[],'score':[]})
expand_table = DataTable(source=expand_table_source, columns=expand_columns, width=500, css_classes=['expand_table'])
phrases_list = MultiSelect(title="", value=[],options=[], width=300, size=27, css_classes=['phrases_list'])
checkbox_group = CheckboxGroup(labels=["Show extracted phrases"], active=[], width=400, css_classes=['checkbox_group'])
search_box_area = column(children=[Div(width=200)])
export_working_label = Div(text="", style={'color':'red'})
getvocab_working_label = Div(text="", style={'color':'blue', 'padding-top': '0px', 'font-size':'15px'})
search_working_label = Div(text="", style={'color':'blue', 'padding-bottom': '0px', 'font-size':'15px'})
expand_working_label = Div(text="", style={'color':'blue', 'padding-top': '7px', 'padding-left':'10px', 'font-size':'15px'})
clear_working_label = Div(text="", style={'color':'blue', 'padding-top': '30px', 'padding-left':'20px', 'font-size':'15px'})
seed_check_label = Div(text='',style={'font-size':'15px'}, height=20, width=500)
seed_layout = Row(seed_input_box,column(Div(height=0, width=0),clear_seed_button), clear_working_label)
table_layout = Row(expand_table,Div(width=25), column(Div(height=350),export_button,export_working_label))
table_area = column(children=[table_layout])
phrases_area = column(children=[search_working_label, Div(width=300)])
checkbox_layout = column(checkbox_group, getvocab_working_label)
grid = layout([
                [Div(width=500), Div(text="<H1>Set Expansion Demo</H1>")],
                [checkbox_layout, seed_layout],
                [search_box_area,Div(width=370), expand_button, expand_working_label],
                [phrases_area, Div(width=100), table_area]
            ])


# define callbacks

def get_phrases(top_n=100000):
    global all_phrases
    received = send_request_to_server('get_vocab')
    # all_phrases.extend(x for x in received if len(x) < max_phrase_length)
    all_phrases=received
    for p in all_phrases:
        if len(p) < max_phrase_length:
            all_phrases_dict[p] = p
            all_cut_phrases_dict[p] = p
        else:
            all_phrases_dict[p] = p[:max_phrase_length] + '...'
            all_cut_phrases_dict[p[:max_phrase_length] + '...'] = p
    print('done. vocab count = ' + str(len(all_phrases)))
    # pp = pprint.PrettyPrinter()
    # pp.pprint(all_phrases_dict)


def send_request_to_server(request):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Connect to server and send data
        sock.connect((expand_host, port))
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

    finally:
        sock.close()


def clean_group(phrase_group):
    text = [x.lstrip() for x in phrase_group.split(';')]
    return min(text, key=len)


def conv(val):
    if val == np.nan:
        return 0 # or whatever else you want to represent your NaN with
    return val


def row_selected_callback(attr, old, new):
    global clear_flag
    if not clear_flag:
        print('row selected callback')
        print('old indices=' + str(old.indices))
        print('new indices=' + str(new.indices))
        global all_selected_phrases
        # selected_rows = new.indices
        # values = ''
        # for x in selected_rows:
        #     values += (expand_table_source.data['res'][x] + ', ')
        # values = values[:-2]


    #sync phrases lists:

    old_phrases = [expand_table_source.data['res'][p] for p in old.indices]
    new_phrases = [expand_table_source.data['res'][p] for p in new.indices]
    print('selected_expand was updated: old=' + str(
        old_phrases) + ' ,new=' + str(new_phrases))
    #phrase was de-selected from expand list:
    for o in old_phrases:
        if o not in new_phrases and all_phrases_dict[o] in phrases_list.value:
            print('removing ' + o + 'from vocab selected')
            phrases_list.value.remove(all_phrases_dict[o])
            break
    #new phrase was selected from expand list:
    for n in new_phrases:
        if n not in old_phrases and all_phrases_dict[n] in phrases_list.options and all_phrases_dict[n] not in phrases_list.value:
            phrases_list.value.append(all_phrases_dict[n])
            break

    update_all_selected_phrases()
    seed_input_box.value = get_selected_phrases_for_seed()


def update_all_selected_phrases():
    print('update selected phrases')
    global all_selected_phrases
    updated_selected_phrases = all_selected_phrases[:]
    selected_expand = [expand_table_source.data['res'][p] for p in expand_table_source.selected.indices]
    selected_vocab = phrases_list.value
    print('selected expand=' + str(selected_expand))
    print('selected vocab=' + str(selected_vocab))
    print('current all_selected_phrases=' + str(all_selected_phrases))
    for x in all_selected_phrases:
        print('x=' + x)
        if (x in expand_table_source.data['res'] and x not in selected_expand) or (all_phrases_dict[x] in phrases_list.options and all_phrases_dict[x] not in selected_vocab):
            print('removing ' + x)
            updated_selected_phrases.remove(x)
    for e in selected_expand:
        if e not in updated_selected_phrases:
            print('adding ' + e)
            updated_selected_phrases.append(e)
    for v in selected_vocab:
        full_v = all_cut_phrases_dict[v]
        if full_v not in updated_selected_phrases:
            print('adding ' + full_v)
            updated_selected_phrases.append(full_v)
    all_selected_phrases = updated_selected_phrases[:]
    print('all_selected_phrases list was updated: ' + str(all_selected_phrases))


def show_phrases_callback(checked_value):
    global search_box_area, phrases_area
    if len(checked_value) == 1:
        if all_phrases is None:
            getvocab_working_label.text = working_text
            get_phrases()
            phrases_list.options = list(all_cut_phrases_dict.keys())[0:max_visible_phrases] #show the cut representation
            getvocab_working_label.text = ''
        search_box_area.children=[search_input_box]
        phrases_area.children=[search_working_label, phrases_list]
    else:
        search_box_area.children=[]
        phrases_area.children=[]


def get_expand_results_callback():
    expand_working_label.text = working_text
    global seed_check_text, table_area
    try:
        seed_check_label.text = ''
        table_area.children = [table_layout]
        seed = seed_input_box.value
        # print('seed= ' + user_input)
        if seed == '':
            expand_table_source.data={'res':[],'score':[]}
            return
        if all_phrases is not None:
            seed_words = [x.strip() for x in seed.split(',')]
            bad_words = ''
            for w in seed_words:
                if w not in all_phrases:
                    bad_words += ("'"+ w + "',")
            if bad_words != '':
                seed_check_label.text = 'the words: <span class="bad-word">' + bad_words[:-1] + '</span> are not in the vocabulary. '
                print('setting table area')
                table_area.children = [seed_check_label,table_layout]
        print('sending expand request to server')
        received = send_request_to_server(seed)
        res = [x[0] for x in received]
        scores = [y[1] for y in received]
        print('setting table data')
        expand_table_source.data = {
            'res': res,
            'score': scores
        }
    except Exception as e:
        print(str(e))
    finally:
        expand_working_label.text = ''



def search_callback(value, old, new):
    search_working_label.text = working_text
    print('search vocab')
    global all_phrases, phrases_list, all_selected_phrases, search_flag
    search_flag = True
    if new == '':
        new_phrases = list(all_cut_phrases_dict.keys())
    else:
        # new_phrases = [x for x in all_phrases if x.startswith(new)]
        new_phrases = [all_phrases_dict[x] for x in all_phrases if (new.lower() in x.lower() or x.lower()==new.lower())]
    phrases_list.options=new_phrases[0:max_visible_phrases]
    phrases_list.value = [all_phrases_dict[x] for x in all_selected_phrases if all_phrases_dict[x] in phrases_list.options]
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
        print('selected_vocab was updated: old=' + str(old_selected) + ' ,new=' + str(new_selected))
        # sync expand table:
        # phrase was de-selected from vocab list:
        expand_selected = [expand_table_source.data['res'][p] for p in expand_table_source.selected.indices]
        for o in old_selected:
            full_o = all_cut_phrases_dict[o]
            if o not in new_selected and full_o in expand_selected:
                print(full_o + ' removed from vocab selected and exists in expand selected')
                print('removing ' + full_o + 'from expand selected indices. index=' + str(expand_table_source.data['res'].index(full_o)))
                print('current expand indices: ' + str(expand_table_source.selected.indices))
                expand_table_source.selected.indices.remove(expand_table_source.data['res'].index(full_o))
                print('new expand indices: ' + str(expand_table_source.selected.indices))
                break
        # new phrase was selected from vocab list:
        for n in new_selected:
            full_n = all_cut_phrases_dict[n]
            print('selected phrase=' + n + ', full phrase=' + full_n)
            if n not in old_selected and full_n in expand_table_source.data['res'] and full_n not in expand_selected:
                expand_table_source.selected.indices.append(expand_table_source.data['res'].index(full_n))
                break
        update_all_selected_phrases()
        seed_input_box.value = get_selected_phrases_for_seed()


def clear_seed_callback():
    print('clear')
    clear_working_label.text = working_text
    global all_selected_phrases, table_area, clear_flag
    clear_flag = True
    seed_input_box.value = ''
    seed_check_label.text = ''
    expand_table_source.selected.indices=[]
    phrases_list.value = []
    all_selected_phrases = []
    # table_area.children = []
    table_area.children = [table_layout]
    clear_flag = False
    clear_working_label.text = ''



def export_data_callback():
    if export_working_label.text != working_text:
        print('saving expansion results to: ' + out_path)
        export_working_label.style = {'color': 'red'}
        export_working_label.text=working_text
        table_df = pandas.DataFrame(expand_table_source.data)
        table_df.to_csv(out_path)
        export_working_label.style={'color':'green'}
        export_working_label.text = 'Done!'
        time.sleep(1)
        export_working_label.text = ''


def get_selected_phrases_for_seed():
    global all_selected_phrases
    phrases = ''
    for x in all_selected_phrases:
        phrases += x + ', '
    phrases = phrases[:-2]
    return phrases


# set callbacks
expand_button.on_click(get_expand_results_callback)
expand_table_source.on_change('selected', row_selected_callback)
checkbox_group.on_click(show_phrases_callback)
search_input_box.on_change('value',search_callback)
phrases_list.on_change('value', vocab_phrase_selected_callback)
clear_seed_button.on_click(clear_seed_callback)
export_button.on_click(export_data_callback)


# arrange components in page
doc = curdoc()
main_title = "Set Expansion Demo"
doc.title = main_title
doc.add_root(grid)


# present initial example:
# get_expand_results_callback()