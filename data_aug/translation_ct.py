from itertools import permutations
from googletrans import Translator
from operator import itemgetter
from bing_translation_for_python import Translator
from bing_trans_ct import *
from baidu_trans_ct import *
import random
import time

LANG_FLAG = 2

LANGUAGES = {
    'af': 'afrikaans','sq': 'albanian','am': 'amharic','ar': 'arabic','hy': 'armenian','az': 'azerbaijani','eu': 'basque','be': 'belarusian','bn': 'bengali',
    'bs': 'bosnian','bg': 'bulgarian','ca': 'catalan','ceb': 'cebuano','ny': 'chichewa','zh-cn': 'chinese (simplified)','zh-tw': 'chinese (traditional)',
    'co': 'corsican','hr': 'croatian','cs': 'czech','da': 'danish','nl': 'dutch','en': 'english','eo': 'esperanto','et': 'estonian','tl': 'filipino',
    'fi': 'finnish','fr': 'french','fy': 'frisian','gl': 'galician','ka': 'georgian','de': 'german','el': 'greek','gu': 'gujarati','ht': 'haitian creole',
    'ha': 'hausa','haw': 'hawaiian','iw': 'hebrew','he': 'hebrew','hi': 'hindi','hmn': 'hmong','hu': 'hungarian','is': 'icelandic','ig': 'igbo',
    'id': 'indonesian','ga': 'irish','it': 'italian','ja': 'japanese','jw': 'javanese','kn': 'kannada','kk': 'kazakh','km': 'khmer','ko': 'korean',
    'ku': 'kurdish (kurmanji)','ky': 'kyrgyz','lo': 'lao','la': 'latin','lv': 'latvian','lt': 'lithuanian','lb': 'luxembourgish','mk': 'macedonian',
    'mg': 'malagasy','ms': 'malay','ml': 'malayalam','mt': 'maltese','mi': 'maori','mr': 'marathi','mn': 'mongolian','my': 'myanmar (burmese)','ne': 'nepali',
    'no': 'norwegian','or': 'odia','ps': 'pashto','fa': 'persian','pl': 'polish','pt': 'portuguese','pa': 'punjabi','ro': 'romanian','ru': 'russian',
    'sm': 'samoan','gd': 'scots gaelic','sr': 'serbian','st': 'sesotho','sn': 'shona','sd': 'sindhi', 'si': 'sinhala','sk': 'slovak','sl': 'slovenian','so': 'somali',
    'es': 'spanish','su': 'sundanese','sw': 'swahili','sv': 'swedish','tg': 'tajik','ta': 'tamil','te': 'telugu','th': 'thai','tr': 'turkish','uk': 'ukrainian',
    'ur': 'urdu','ug': 'uyghur','uz': 'uzbek','vi': 'vietnamese','cy': 'welsh','xh': 'xhosa','yi': 'yiddish','yo': 'yoruba','zu': 'zulu'}

BING_LANGUAGES = {
    'ar':'Arabic','ca':'Catalan','zh-Hans':'Chinese (Simplified)','zh-Hant':'Chinese (Traditional)','hr':'Croatian','en':'English',
    'fr':'French','de':'German','el':'Greek','he':'Hebrew','hi':'Hindi','it':'Italian','ja':'Japanese','ko':'Korean',
    'pt':'Portuguese','ru':'Russian','es':'Spanish','th':'Thai','tr':'Turkish','vi':'Vietnamese',
}

BAIDU_LANGUAGES = {
    0:'kor', 1:'th', 2:'pt', 3:'el', 4:'bul', 5:'fin', 6:'slo', 7:'zh', 8:'fra', 9:'ara', 10:'de',
    11:'nl', 12:'est', 13:'cs', 14:'swe', 15:'vie', 16:'en', 17:'jp', 18:'spa', 19:'ru', 20:'it',
    21:'pl', 22:'dan', 23:'rom', 24:'hu'
}

# 翻译单句
def trans_str_google(string, dest):
    # translator = Translator(service_urls = 'https://translate.google.cn/')
    translator = Translator(service_urls = 'https://translate.google.cn/')
    res = translator.translate(string, dest)
    return res.text

def trans_str_bing(string, dest, src_lang = 'en'):
    res = trans_bing(string, src_lang, dest)
    # print(type(res))
    # print(res)
    return res[0]['translations'][0]['text']

def trans_str_baidu(string, dest, src_lang = 'en'):
    # 把每次调用拉长到1s
    start = time.time()
    res = trans_baidu(string, src_lang, dest)
    end = time.time()
    if end-start < 1:
        time.sleep(1.1-end+start)
    if type(res) == int:
        if res == 52001 or res == 52002:
            res = trans_str_baidu(string, dest, src_lang)
            return res
        elif res == 54005:
            time.sleep(3)
            res = trans_str_baidu(string, dest, src_lang)
            return res
        else:
            print(res)
            return -1
    else:
        return res
            
# cnt种中间语言的全排列 para_num：样本数 list_len：每个样本需要生成的样本数
def trans_chain_baidu(string, lst, cnt, src_lang, dic, para_num, list_len=-1):
    res = []
    # generate permutations (A lst cnt)
    idx = list(permutations(lst, cnt))
    random.shuffle(idx)
    length = len(idx)

    for i in range(length):
        seq = idx[i]
        start_lang = dic[seq[0]]
        end_lang = dic[seq[-1]]
        # print(seq)
        if start_lang != src_lang and end_lang != src_lang:
           # map number to lang sequence
           seq = itemgetter(*seq)(dic)
           # chain of translation
           temp_lang = 'en'
           res_string = string
           for j in range(len(seq)):
               lang = seq[j]
               res_string = trans_str_baidu(res_string, lang, temp_lang)
               temp_lang = lang
           # result
           res_string = trans_str_baidu(res_string, src_lang, temp_lang)
           if (len(res) < list_len * para_num and list_len != -1) or (list_len == -1):
                res.extend(res_string)
                # print(res_string)
           else:
               break
    # print(len(res))
    return res

# cnt种中间语言的全排列
def trans_chain_bing(string, lst, cnt, src_lang, dic, list_len=-1):
    res = []
    # generate permutations (A lst cnt)
    idx = list(permutations(lst, cnt))
    random.shuffle(idx)
    length = len(idx)

    for i in range(length):
        seq = idx[i]
        start_lang = dic[seq[0]]
        end_lang = dic[seq[-1]]
        # print(seq)
        if start_lang != src_lang and end_lang != src_lang:
           # map number to lang sequence
           seq = itemgetter(*seq)(dic)
           # chain of translation
           temp_lang = 'en'
           res_string = string
           for j in range(len(seq)):
               lang = seq[j]
               res_string = trans_str_bing(res_string, lang, temp_lang)
               temp_lang = lang
           # result
           res_string = trans_str_bing(res_string, src_lang, temp_lang)
           if (len(res) < list_len and list_len != -1) or (list_len == -1):
                res.append(res_string)
                print(res_string)
           else:
               break
    # print(len(res))
    return res

# 选定一种语言来回翻译（反复横跳）
def trans_round_bing(string, lst, cnt, src_lang, dic, lang = -1):
    if lang == -1: # no specific language
        idx = list(permutations(lst, 1))
        random.shuffle(idx)
        lang = dic[idx[0][0]]
        if lang == 'en':
            lang = dic[idx[1][0]]
    res_string = string
    for i in range(cnt):
        res_string = trans_str_bing(res_string, lang, src_lang)
        res_string = trans_str_bing(res_string, src_lang, lang)
    return  res_string

# 每一次语言都不同
def trans_round_bing_ex(string, lst, cnt, src_lang, dic, lang = -1):
    res = []
    for i in range(cnt):
        res_string = trans_round_bing(string, lst, i+5, src_lang, dic, lang)
        res.append(res_string)
    return res


# if __name__ == '__main__':
    
#     # generate new dict
#     if LANG_FLAG == 1:
#         idx = list(range(len(BING_LANGUAGES)))
#         keys = list(BING_LANGUAGES.keys())
#         langs = dict(zip(idx,keys))
#     elif LANG_FLAG == 2:
#         langs = BAIDU_LANGUAGES
#     else:
#         idx = list(range(len(LANGUAGES)))
#         keys = list(LANGUAGES.keys())
#         langs = dict(zip(idx,keys))

#     string = "This paper addresses the following question: does a small, essential, core set of API members emerges from the actual usage of the API by client applications? To investigate this question, we study the 99 most popular libraries available in Maven Central and the 865,560 client programs that declare dependencies towards them, summing up to 2.3M dependencies. Our key findings are as follows: 43.5% of the dependencies declared by the clients are not used in the bytecode; all APIs contain a large part of rarely used types and a few frequently used types, and the ratio varies according to the nature of the API, its size and its design; we can systematically extract a reuse-core from APIs that is sufficient to provide for most clients, the median size of this subset is 17% of the API that can serve 83% of the clients. This study is novel both in its scale and its findings about unused dependencies and the reuse-core of APIs. Our results provide concrete insights to improve Maven's build process with a mechanism to detect unused dependencies. They also support the need to reduce the size of APIs to facilitate API learning and maintenance.  "
#     res = trans_round_bing_ex(string, idx, 5, 'en', langs)
#     # res = trans_chain(string, idx, 3, 'en', langs, 10)
#     for i in range(len(res)):
#         print(res[i])


