import pandas as pd
from pandas.core.indexes import category

label_dict = {'cs.CL': 0, 'cs.NE': 1, 'cs.DL': 2, 'cs.CV': 3, 'cs.LG': 4, 'cs.DS': 5, 'cs.IR': 6, 'cs.RO': 7, 'cs.DM': 8, 'cs.CR': 9, 'cs.AR': 10, 'cs.NI': 11, 'cs.AI': 12, 'cs.SE': 13, 'cs.CG': 14, 'cs.LO': 15, 'cs.SY': 16, 'cs.GR': 17, 'cs.PL': 18, 'cs.SI': 19, 'cs.OH': 20, 'cs.HC': 21, 'cs.MA': 22, 'cs.GT': 23, 'cs.ET': 24, 'cs.FL': 25, 'cs.CC': 26, 'cs.DB': 27, 'cs.DC': 28, 'cs.CY': 29, 'cs.CE': 30, 'cs.MM': 31, 'cs.NA': 32, 'cs.PF': 33, 'cs.OS': 34, 'cs.SD': 35, 'cs.SC': 36, 'cs.MS': 37, 'cs.GL': 38}
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]
key = get_key(label_dict, 2)[0]
print(type(key))

sub = pd.read_csv("./processed_data/submission_unprocessed.csv")
category_list = []
for label_num in sub["label_num"]:
    key = get_key(label_dict, int(label_num))[0]
    category_list.append(key)
sub["categories"] = category_list
sub = sub.drop(['text','label_num'], axis=1)
sub.to_csv("./processed_data/submission_processed.csv", index=0,encoding='utf-8')
# sub = pd.read_csv("./processed_data/submission_processed.csv",index_col=0)
# sub.to_csv("./processed_data/submission_processed.csv")
# print(category_list)
