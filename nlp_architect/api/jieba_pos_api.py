# -*- coding: utf-8 -*-

"""
Created on 2018/10/6 下午7:04

@author: xujiang@baixing.com

"""

import jieba
import jieba.posseg as psg
from nlp_architect.api.abstract_api import AbstractApi

dic = {
"a": "形容词",
"ad": "副形词",
"ag": "形语素",
"an": "名形词",
"b": "区别词",
"c": "连词",
"d": "副词",
"dg": "副语素",
"e": "叹词",
"f": "方位词",
"g": "语素",
"h": "前接成分",
"i": "成语",
"j": "简称略语",
"k": "后接成分",
"l": "习用语",
"m": "数词",
"n": "名词",
"ng": "名语素",
"nr": "人名",
"ns": "地名",
"nt": "机构团体",
"nx": "字母专名",
"nz": "其他专名",
"o": "拟声词",
"p": "介词",
"q": "量词",
"r": "代词",
"s": "处所词",
"t": "时间词",
"tg": "时语素",
"u":"助词",
"ud":"结构助词",
"ug": "时态助词",
"uj":"结构助词的",
"ul":"时态助词了",
"uv":"结构助词地",
"uz":"时态助词着",
"v":"动词",
"vd":"副动词",
"vg":"动语素",
"vn":"名动词",
"w":"标点符号",
"x":"非语素字",
"y":"语气词",
"z":"状态词"
}



class JiebaPosApi(AbstractApi):
    def __init__(self):
        self.model = None

    def load_model(self):
        """
        Load spacy english model
        """
        pass
    
    def pretty_print(self, seg_list):
        ret = {}
        spans = []
        counter = 0
        words = []
        tags = []
        for word, tag in seg_list:
            tag = dic.get(tag, '其他')
            words.append(word)
            tags.append(tag)
            spans.append({
                'start': counter,
                'end': (counter + len(word)),
                'type': tag
            })
            counter += len(word) + 1
        ret['doc_text'] = ' '.join(words)
        ret['annotation_set'] = tags
        ret['spans'] = spans
        ret['title'] = 'None'
        print (ret)
        return {"doc": ret, 'type': 'high_level'}

    def inference(self, doc):
        """
        Parse according to SpacyNer's model

        Args:
            doc (str): the doc str

        Returns:
            :obj:`nlp_architect.utils.high_level_doc.HighLevelDoc`: the model's response hosted in
                HighLevelDoc object
        """
        print(doc)
        seg_list = psg.cut(doc)
        return self.pretty_print(seg_list)
