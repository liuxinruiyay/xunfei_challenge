import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import warnings
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




# Training Function
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
            output = model(titletext, labels = labels, attention_mask = attention_mask)
            
            loss = output.loss
            
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
            output = model(titletext, labels = labels, attention_mask = attention_mask)

            logits = output.logits
            y_pred.extend(torch.argmax(logits, 1).tolist())
            y_true.extend(labels.tolist())

        # evaluation
        print('Classification Report:')
        print(classification_report(y_true, y_pred))
        tmp_valid_acc = accuracy_score(y_true, y_pred)

        # print progress
        print('Epoch [{}/{}],Train Loss: {:.4f},Valid Accuracy: {:.4f}'
                      .format(epoch+1, num_epochs, running_loss, tmp_valid_acc))

        # checkpoint
        if best_valid_acc < tmp_valid_acc:
            best_valid_acc = tmp_valid_acc
            save_checkpoint(file_path + '/'+'model-'+str(best_valid_acc)+'.pt', model, best_valid_acc)
    
    print('Finished Training!')

