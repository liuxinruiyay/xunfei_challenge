import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
from Focal_Loss import FocalLoss
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Save and Load Functions
def save_checkpoint(save_path, model, valid_acc):
    if save_path == None:
        return  
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_acc': valid_acc}  
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):  
    if load_path==None:
        return 
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return  
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}  
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path==None:
        return  
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')  
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']




# Basical Training Function
def train(model, optimizer, train_loader, valid_loader,num_epochs = 5, file_path='./pretrain', best_valid_acc = 0.0):
    model = model.to(device)
    # training loop   
    for epoch in range(num_epochs):
        print("Epoch:{}".format(str(epoch+1)))
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            output = model(texts=titletext, labels=labels, attn_mask=attention_mask)
            
            loss = output[0]
            
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
        print('loss:{:.4f}'.format(running_loss))
        # evaluation step
        y_pred = []
        y_true = []
        model.eval()                    
        # validation loop
        for batch in valid_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(texts=titletext, labels=labels, attn_mask=attention_mask)

            logits = output[1]
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())

        # evaluation
        print('Classification Report:')
        print(classification_report(y_true, y_pred))
        tmp_valid_acc = accuracy_score(y_true, y_pred)
        tmp_f1_score = f1_score(y_true, y_pred, average='macro')

        # print progress
        print('Epoch [{}/{}],Train Loss: {:.4f},Valid Accuracy: {:.4f}, f1 score: {:.4f}'
                      .format(epoch+1, num_epochs, running_loss, tmp_valid_acc, tmp_f1_score))

        # checkpoint
        if best_valid_acc < tmp_valid_acc:
            best_valid_acc = tmp_valid_acc
            save_checkpoint(file_path + '/'+'model-'+str(best_valid_acc)+'.pt', model, best_valid_acc)
    
    print('Finished Training!')

# Training Function
def train_with_focal_loss(model, optimizer, train_loader, valid_loader,num_epochs = 5, file_path='./pretrain', best_valid_acc = 0.0):
    model = model.to(device)
    alpha_list = [0.2,1.2,1.8,0.1,6.0,0.4,1.2,0.5,18.0,0.4,22.0,0.3,0.4,0.5,1.3,0.5,0.6,2.3,1.1,1.5,13.0,1.0,26.0,1.0,2.5,2.0,12.0,1.0,0.5,6.0,2.2,2.5,2.0,35.0,4.1,3.1,3.0,40.0,200.0]
    alpha_list = torch.tensor(alpha_list, dtype=torch.long)
    loss_fn = FocalLoss(alpha=alpha_list, gamma=2, class_num=39)
    # training loop   
    for epoch in range(num_epochs):
        print("Epoch:{}".format(str(epoch+1)))
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            #print(attention_mask)
            optimizer.zero_grad()
            output = model(texts=titletext, labels=labels, attn_mask=attention_mask)
            
            #loss = output[0]
            #使用focal_loss计算损失函数
            preds = output[1]
            # print(preds)
            # print(labels)
            loss = loss_fn(inputs=preds, targets=labels)
            # print(loss)
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
        print('loss:{:.4f}'.format(running_loss))
        # evaluation step
        y_pred = []
        y_true = []
        model.eval()                    
        # validation loop
        for batch in valid_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(texts=titletext, labels=labels, attn_mask=attention_mask)

            logits = output[1]
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())

        # evaluation
        print('Classification Report:')
        print(classification_report(y_true, y_pred))
        tmp_valid_acc = accuracy_score(y_true, y_pred)
        tmp_f1_score = f1_score(y_true, y_pred, average='macro')

        # print progress
        print('Epoch [{}/{}],Train Loss: {:.4f},Valid Accuracy: {:.4f}, f1 score: {:.4f}'
                      .format(epoch+1, num_epochs, running_loss, tmp_valid_acc, tmp_f1_score))

        # checkpoint
        if best_valid_acc < tmp_valid_acc:
            best_valid_acc = tmp_valid_acc
            save_checkpoint(file_path + '/'+'model-'+str(best_valid_acc)+'.pt', model, best_valid_acc)
    
    print('Finished Training!')

# Training Function
def train_with_kfold(k_index, model, optimizer, train_loader, valid_loader,num_epochs = 5, file_path='./pretrain', best_valid_acc = 0.0):
    model = model.to(device)
    # training loop   
    for epoch in range(num_epochs):
        print("Epoch:{}".format(str(epoch+1)))
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            output = model(texts=titletext, labels=labels, attn_mask=attention_mask)
            
            loss = output[0]
            
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
        print('loss:{:.4f}'.format(running_loss))
        # evaluation step
        y_pred = []
        y_true = []
        model.eval()                    
        # validation loop
        for batch in valid_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(texts=titletext, labels=labels, attn_mask=attention_mask)

            logits = output[1]
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())

        # evaluation
        print('Classification Report:')
        print(classification_report(y_true, y_pred))
        tmp_valid_acc = accuracy_score(y_true, y_pred)
        tmp_f1_score = f1_score(y_true, y_pred, average='macro')

        # print progress
        print('Epoch [{}/{}],Train Loss: {:.4f},Valid Accuracy: {:.4f}, f1 score: {:.4f}'
                      .format(epoch+1, num_epochs, running_loss, tmp_valid_acc, tmp_f1_score))

        # checkpoint
        if best_valid_acc < tmp_valid_acc:
            best_valid_acc = tmp_valid_acc
            save_checkpoint(file_path + '/'+str(k_index)+'-'+'model-'+str(best_valid_acc)+'.pt', model, best_valid_acc)
    
    print('Finished Training!')


def train_use_lstm(model, optimizer, train_loader, valid_loader,num_epochs = 5, file_path='./pretrain', best_valid_acc = 0.0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # training loop   
    for epoch in range(num_epochs):
        print("Epoch:{}".format(str(epoch+1)))
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            output = model(texts=titletext, attn_mask=attention_mask)
            
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
        print('loss:{:.4f}'.format(running_loss))
        # evaluation step
        y_pred = []
        y_true = []
        model.eval()                    
        # validation loop
        for batch in valid_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(texts=titletext, attn_mask=attention_mask)

            logits = output
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())

        # evaluation
        print('Classification Report:')
        print(classification_report(y_true, y_pred))
        tmp_valid_acc = accuracy_score(y_true, y_pred)
        tmp_f1_score = f1_score(y_true, y_pred, average='macro')

        # print progress
        print('Epoch [{}/{}],Train Loss: {:.4f},Valid Accuracy: {:.4f}, f1 score: {:.4f}'
                      .format(epoch+1, num_epochs, running_loss, tmp_valid_acc, tmp_f1_score))

        # checkpoint
        if best_valid_acc < tmp_valid_acc:
            best_valid_acc = tmp_valid_acc
            save_checkpoint(file_path + '/'+'model-'+str(best_valid_acc)+'.pt', model, best_valid_acc)
    
    print('Finished Training!')

# Training Function
def train_with_kfold_lstm(k_index, model, optimizer, train_loader, valid_loader,num_epochs = 5, file_path='./pretrain', best_valid_acc = 0.0):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # training loop   
    for epoch in range(num_epochs):
        print("Epoch:{}".format(str(epoch+1)))
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            output = model(texts=titletext, attn_mask=attention_mask)
            
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
        print('loss:{:.4f}'.format(running_loss))
        # evaluation step
        y_pred = []
        y_true = []
        model.eval()                    
        # validation loop
        for batch in valid_loader:
            labels = batch['targets'].to(device)
            titletext = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(texts=titletext, attn_mask=attention_mask)

            logits = output
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())

        # evaluation
        print('Classification Report:')
        print(classification_report(y_true, y_pred))
        tmp_valid_acc = accuracy_score(y_true, y_pred)
        tmp_f1_score = f1_score(y_true, y_pred, average='macro')

        # print progress
        print('Epoch [{}/{}],Train Loss: {:.4f},Valid Accuracy: {:.4f}, f1 score: {:.4f}'
                      .format(epoch+1, num_epochs, running_loss, tmp_valid_acc, tmp_f1_score))

        # checkpoint
        if best_valid_acc < tmp_valid_acc:
            best_valid_acc = tmp_valid_acc
            save_checkpoint(file_path + '/'+str(k_index)+'-'+'model-'+str(best_valid_acc)+'.pt', model, best_valid_acc)
    
    print('Finished Training!')