import requests
import random
import json
from hashlib import md5
import time

# Set your own appid/appkey.
appid = '20210724000896953'
appkey = '7RQgx8wj_pLW7E_Aqwtq'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang =  'zh'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

# 段与段间用\n分开，每次请求最多送三个样本（三段）
def trans_baidu(string, src_lang, dst_lang):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + string + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': string, 'from': src_lang, 'to': dst_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    # res = json.dumps(result, indent=4, ensure_ascii=False)
    # print(res)
    
    # 判断是否有error_code
    if "error_code" in result.keys():
        res = result['error_code']
        return int(res)
    else:
        list_len = len(result['trans_result'])
        lst = []
        for i in range(list_len):
            sub_string = result['trans_result'][i]['dst']
            lst.append(sub_string)
        return lst


# if __name__ == "__main__":
#     string = "An Easy-to-use Real-world Multi-objective Optimization Problem Suite	 \
#     Although synthetic test problems are widely used for the performance \
#     assessment of evolutionary multi-objective optimization algorithms, they are \
#     likely to include unrealistic properties which may lead to \
#     overestimation/underestimation. To address this issue, we present a \
#     multi-objective optimization problem suite consisting of 16 bound-constrained \
#     real-world problems. The problem suite includes various problems in terms of \
#     the number of objectives, the shape of the Pareto front, and the type of design \
#     variables. 4 out of the 16 problems are multi-objective mixed-integer \
#     optimization problems. We provide Java, C, and Matlab source codes of the 16 \
#     problems so that they are available in an off-the-shelf manner. We examine an \
#     approximated Pareto front of each test problem. We also analyze the performance \
#     of six representative evolutionary multi-objective optimization algorithms on \
#     the 16 problems. In addition to the 16 problems, we present 8 constrained \
#     multi-objective real-world problems.\n\
#     Exploration of reproducibility issues in scientometric research Part 1: \
#     Direct reproducibility This is the first part of a small-scale explorative study in an effort to \
#     start assessing reproducibility issues specific to scientometrics research. \
#     This effort is motivated by the desire to generate empirical data to inform \
#     debates about reproducibility in scientometrics. Rather than attempt to \
#     reproduce studies, we explore how we might assess ""in principle"" \
#     reproducibility based on a critical review of the content of published papers. \
#     The first part of the study focuses on direct reproducibility - that is the \
#     ability to reproduce the specific evidence produced by an original study using \
#     the same data, methods, and procedures. The second part (Velden et al. 2018) is \
#     dedicated to conceptual reproducibility - that is the robustness of knowledge \
#     claims towards verification by an alternative approach using different data, \
#     methods and procedures. The study is exploratory: it investigates only a very \
#     limited number of publications and serves us to develop instruments for \
#     identifying potential reproducibility issues of published studies: These are a \
#     categorization of study types and a taxonomy of threats to reproducibility. We \
#     work with a select sample of five publications in scientometrics covering a \
#     variation of study types of theoretical, methodological, and empirical nature. \
#     Based on observations made during our exploratory review, we conclude this \
#     paper with open questions on how to approach and assess the status of direct \
#     reproducibility in scientometrics, intended for discussion at the special track \
#     on ""Reproducibility in Scientometrics"" at STI2018 in Leiden.\n\
#     Scheduled Sampling for Transformers	Scheduled sampling is a technique for avoiding one of the known problems in \
#     sequence-to-sequence generation: exposure bias. It consists of feeding the \
#     model a mix of the teacher forced embeddings and the model predictions from the \
#     previous step in training time. The technique has been used for improving the \
#     model performance with recurrent neural networks (RNN). In the Transformer \
#     model, unlike the RNN, the generation of a new word attends to the full \
#     sentence generated so far, not only to the last word, and it is not \
#     straightforward to apply the scheduled sampling technique. We propose some \
#     structural changes to allow scheduled sampling to be applied to Transformer \
#     architecture, via a two-pass decoding strategy. Experiments on two language \
#     pairs achieve performance close to a teacher-forcing baseline and show that \
#     this technique is promising for further exploration."
#     start = time.time()
#     lst = trans_baidu(string, 'en', 'zh')
#     end = time.time()
#     print("运行时间:%.2f秒"%(end-start))
#     print(lst)