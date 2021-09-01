# 学术论文文本分类
1.Baseline model: BertForSequenceClassification  
2.Trick1:使用imblearn对文本进行上采样和下采样  
3.Trick2:使用Focal loss解决样本分布不均衡问题   
![image](https://user-images.githubusercontent.com/38974623/131627876-9f4fc634-e2cf-402c-9c73-d7fcc2103e08.png)   
![image](https://user-images.githubusercontent.com/38974623/131627879-de8b6640-4d70-4c5d-8cfe-32c1d8bfce38.png)   
![image](https://user-images.githubusercontent.com/38974623/131627894-534fafd8-546c-48c5-8066-2a8461939a62.png)   
![image](https://user-images.githubusercontent.com/38974623/131631197-548a9009-cd47-4110-be0a-cf4ecead6366.png)

4.Trick3:使用Bert的预训练embedding，叠加一个双层LSTM，再接一个全连接输出logits   
5.Trick4:使用数据增强技术：baidu翻译，必应翻译对文本进行反复翻译增强文本   
6.Trick5:使用文本清洗方法,去除原有文本中的停用词   
7.Trick6:使用KFold进行10折交叉验证，对每个模型输出的logits取加权平均   
