# 学术论文文本分类
1.Baseline model: BertForSequenceClassification
2.Trick1:使用imblearn对文本进行上采样和下采样
3.Trick2:使用Focal loss解决样本分布不均衡问题
4.Trick3:使用Bert的预训练embedding，叠加一个双层LSTM，再接一个全连接输出logits
5.Trick4:使用数据增强技术：baidu翻译，必应翻译对文本进行反复翻译增强文本
5.Trick5:使用文本清洗方法,去除原有文本中的停用词
6.Trick6:使用KFold进行10折交叉验证，对每个模型输出的logits取加权平均
