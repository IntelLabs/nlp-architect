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

from __future__ import unicode_literals, print_function, division, \
    absolute_import

import argparse
import csv
import os

from .data import absolute_path

from nlp_architect.utils.io import validate_existing_directory

tratz2011_train_labeled_dict = {
    False: [55, 64, 67, 68, 100, 104, 121, 150, 444, 492, 782, 798, 878, 942, 952, 967, 990, 1012,
            1031, 1036, 1371, 1658, 1679, 1717, 1719, 2845, 2904, 3454, 3921, 4040, 4059, 4123,
            4334, 4435, 4512, 4834, 4890, 4902, 4919, 4923, 4925, 4926, 4932, 4988, 5004, 5078,
            5081, 5083, 5094, 5197, 5208, 5246, 5315, 5342, 5409, 5440, 5446, 5656, 5678, 5684,
            5688, 5713, 5719, 5723, 5764, 5776, 5779, 5848, 6038, 6049, 6115, 6116, 6136, 6140,
            6209, 6236, 6297, 6331, 6334, 6471, 6478, 6490, 6731, 6734, 6736, 6737, 6740, 6798,
            6811, 6836, 6837, 6841, 6847, 6848, 6853, 6899, 6900, 6940, 6947, 7003, 7011, 7101,
            7107, 7125, 7189, 7330, 7393, 7400, 7500, 7504, 7512, 7591, 7594, 7615, 7742, 7765,
            7842, 7935, 8546, 8613, 8621, 8642, 8658, 8665, 8741, 8792, 8823, 8863, 8876, 8878,
            9169, 9210, 9277, 9280, 9341, 9403, 9435, 9483, 9517, 9600, 9697, 9749, 9807, 9818,
            9842, 9906, 10098, 10161, 10194, 10273, 10315, 10350, 10351, 10396, 10441, 10463,
            10468, 10490, 10492, 10497, 10509, 10510, 10537, 10581, 10589, 10616, 10737, 10751,
            10922, 10944, 10960, 11007, 11013, 11021, 11181, 11188, 11255, 11297, 11322, 11444,
            11464, 11466, 11469, 11474, 11499, 11608, 11629, 11636, 11679, 11692, 11747, 11792,
            11865, 11894, 11898, 11908, 12032, 12045, 12046, 12067, 12109, 12173, 12207, 12222,
            12385, 12386, 12398, 12408, 12472, 12556, 12669, 12677, 12679, 12755, 12788, 12809,
            12818, 12822, 12844, 13031, 13041, 13120, 13122, 13127, 13147, 13160, 13161, 13186,
            13188, 13189, 13194, 13214, 13236, 13318, 13368, 13421, 13453, 13456, 13505, 13535,
            13583, 13584, 13611, 13653, 13691, 13692, 13701, 13737, 13797, 14007, 14029, 14047,
            14066, 14069, 14092],
    True: [60, 207, 247, 258, 268, 362, 456, 485, 641, 694, 1025, 1272, 1304, 1317, 1542, 1602,
           1643, 1746, 1908, 1909, 2028, 2302, 3424, 3627, 4113, 4398, 4399, 4542, 4759, 4836,
           4849, 4957, 5041, 5126, 5306, 5630, 5661, 5708, 5791, 5971, 5983, 6142, 6548, 7293,
           7416, 7923, 8932, 9700, 10486, 10746, 10803, 11448, 11781, 12072, 12308, 12354, 12368,
           12470, 12510, 12647, 12662, 12766, 12821, 12879, 13494, 14014, 14018, 14020, 14091]
}

tratz2011_val_labeled_dict = {
    False: [12, 16, 17, 18, 19, 21, 23, 24, 25, 26, 27, 28, 29, 32, 40, 42, 43, 45, 47, 48],
    True: [33, 121, 240, 365, 425]
}


def rebuild_row(lst, is_collocation):
    """
    Re-construct csv row as expected by data.py `read_csv_file_data` method

    Args:
        lst (list(str)): list with a string containing '\t'
        is_collocation (bool): if collocation True, else False

    Returns:
        list(str): list of string where the first entry is the noun phrase and the second is 0\1
    """
    split_list = lst[0].split('\t')
    if is_collocation:
        return [split_list[0] + ' ' + split_list[1], '1']
    return [split_list[0] + ' ' + split_list[1], '0']


def read_from_tratz_2011(file_full_path, labeled_dict):
    """
    Read tratz_2011 files and print re-formatted csv files

    Args:
        file_full_path (str): file path
        labeled_dict (dict): dictionary with prepered labels
    """
    # 1. read the data
    with open(file_full_path, 'r', encoding='utf-8-sig') as input_file:
        reader = csv.reader((line.replace('\0', '') for line in input_file))
        reader_list = list(reader)
        csv_data = []
        for index, row in enumerate(reader_list):
            if index in labeled_dict[False]:
                csv_data.append(rebuild_row(row, False))
            if index in labeled_dict[True]:
                csv_data.append(rebuild_row(row, True))
        # 2. write to csv file
        write_csv(csv_data, file_full_path)


def write_csv(data, output):
    """
    Write csv data

    Args:
        output (str): output file path
        data (list(str)):
            the csv formated data
    """
    output_path = output[:-3] + 'csv'
    with open(output_path, 'w', encoding='utf-8') as out_file:
        writer = csv.writer(out_file, delimiter=',', quotechar='"')
        print("CSV file is saved in {0}".format(output_path))
        for result_row in data:
            writer.writerow(result_row)


def preprocess_tratz_2011(folder_path):
    """
    Pre-process tratz_2011 dataset

    Args:
        folder_path (str): path to the unzipped tratz_2011 dataset
    """
    files = ['tratz2011_coarse_grained_random/train.tsv', 'tratz2011_coarse_grained_random/'
                                                          'val.tsv']
    dicts = [tratz2011_train_labeled_dict, tratz2011_val_labeled_dict]
    # 1. get abs path
    if not os.path.isabs(folder_path):
        # handle case using default value\relative paths
        folder_path = os.path.join(os.path.dirname(__file__), folder_path)
    # 2. add the location of the train file in the folder
    for file, dic in zip(files, dicts):
        file_full_path = os.path.join(folder_path, file)
        read_from_tratz_2011(file_full_path, dic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process Tratz 2011 data from tsv to csv')
    parser.add_argument('--data', type=validate_existing_directory,
                        help='path the Tratz_2011_dataset folder local path')
    args = parser.parse_args()
    data_path = absolute_path(args.data)
    preprocess_tratz_2011(data_path)
