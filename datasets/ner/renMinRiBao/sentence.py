#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Time    : 2018/9/20 23:32

__author__ = 'xujiang@baixing.com'

#encoding=utf8
from enum import Enum

"""
输出语料的标签前缀
该类主要是原始语料标签与输出语料标签的映射关系
"""
class TagPrefix(Enum):
    general = '' # 非命名实体的标记
    t = 'Date_'  # 时间类型的标记

    @classmethod
    def convert(cls):
        dicTag = {}
        for name, member in TagPrefix.__members__.items():
            dicTag[name] = member.value

        return dicTag

"""
输出语料的标签后缀 BMES 标注体系
"""
class TagSurfix(Enum):
    S = 's'
    B = 'b'
    M = 'm'
    E = 'e'


class Sentence:
    def __init__(self):
        self.tokens = [] # token
        self.tags = [] # token对应的类型
        self.chars = 0

    def addToken(self, t, tag):
        self.chars += len(t)
        self.tokens.append(t)
        self.tags.append(tag)

    def clear(self):
        self.tokens = []
        self.chars = 0
        self.tags = []


    """
      按照字符拆分token列表中的每一个token
      其中x里面存储的是token的字符序列, y中存储的是相关序列对应的标记
    """
    def generate_tr_line(self, x, y):
        for idx in range(len(self.tokens)):
            t = self.tokens[idx]
            tagstr = self.tags[idx]
            if len(t) == 1:
                x.append(t[0])
                y.append(tagstr + TagSurfix.S.value)
            else:
                nn = len(t)
                for i in range(nn):
                    x.append(t[i])
                    if i == 0:
                        y.append(tagstr + TagSurfix.B.value)
                    elif i == (nn - 1):
                        y.append(tagstr + TagSurfix.E.value)
                    else:
                        y.append(tagstr + TagSurfix.M.value)