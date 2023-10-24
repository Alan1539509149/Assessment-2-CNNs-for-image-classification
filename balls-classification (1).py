
import time
from collections import Counter
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score,confusion_matrix,f1_score,precision_score,classification_report,roc_curve,auc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


train_path = '/kaggle/input/balls-image-classification/train'

train_dataset = datasets.ImageFolder(root=train_path)
train_length=len(train_dataset)



all_labels=[]

for i in range(train_length):
    v=train_dataset[i][1]
    all_labels.append(v)

labels= Counter(all_labels).keys()
counts = Counter(all_labels).values()


plt.bar(labels, counts)


transform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
valid_folder='/kaggle/input/balls-image-classification/valid'
test_folder='/kaggle/input/balls-image-classification/test'
dataset_trans = datasets.ImageFolder(root=train_path, transform=transform)
dataset_valid = datasets.ImageFolder(root=valid_folder, transform=transform)
dataset_test = datasets.ImageFolder(root=test_folder, transform=transform1)

batch_size = 64

train_dataset_loader = DataLoader(dataset_trans, batch_size=batch_size, shuffle=True)
valid_dataset_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
test_dataset_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

dataloaders = {'train': train_dataset_loader, 'valid': valid_dataset_loader}



def model_training(model, dataloaders, criterion, optimizer, device, num_epochs=50):
    model.to(device)

    valid_accuracy,train_accuracy,loss_List1,loss_list2 = [],[],[],[]

    for epoch in range(num_epochs):
        str1='Training Epoch {}/{}'.format(epoch, num_epochs - 1)
        print(str1)

        for i in ['train', 'valid']:
            if i != 'train':
                model.eval()
            else:
                model.train()
            running_loss = 0.0
            running_corrects = 0

            loader=dataloaders[i]
            for inputs, labels in loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

           
                optimizer.zero_grad()
            

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

               
                if i != 'valid':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[i].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[i].dataset)
            

            str1=str(i)+'Training loss: '+str(round(epoch_loss,4))+' Training ACC: '+str(round(epoch_acc,4))
            print(str1)
            
            if i != 'train':
                valid_accuracy.append(epoch_acc)
                loss_list2.append(epoch_loss)

            if i != 'valid':
                train_accuracy.append(epoch_acc)
                loss_List1.append(epoch_loss)
            

    return model, valid_accuracy, train_accuracy,loss_list2, loss_List1



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Residual_Block(nn.Module):
    def __init__(self,in_chanel,out_chanel,s,p):
        super(Residual_Block, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_chanel, out_chanel, 3,stride=s, padding = p),nn.BatchNorm2d(out_chanel),
                                nn.ReLU(inplace=True),nn.Conv2d(out_chanel, out_chanel, 3, stride=s,padding = p),
                                nn.BatchNorm2d(out_chanel) )
        self.residual = nn.Sequential()
        if (in_chanel != out_chanel):
            self.residual = nn.Sequential(nn.Conv2d(in_chanel, out_chanel, 3, stride=s, padding = p), nn.BatchNorm2d(out_chanel))
        self.relu= nn.ReLU(inplace=True)
    def forward(self,x):
      out=self.conv(x)
      out += self.residual(x)
      out = self.relu(out)
      return out    
class ResNet_basic(nn.Module):
    
    def __init__(self,in_chanel,hidden_chanels,out_chanel):
        super(ResNet_basic, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_chanel, 64, 7,stride = 2,padding=3),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                nn.MaxPool2d(3,stride=2,padding=1))

        p=hidden_chanels[0]

        p1,p2,p3,p4=p[0],p[1],p[2],p[3]
        self.res1 = Residual_Block(p1,p2,p3,p4)

        p = hidden_chanels[1]
        p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
        self.res2 = Residual_Block(p1, p2, p3, p4)

        p = hidden_chanels[2]
        p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
        self.res3 = Residual_Block(p1, p2, p3, p4)

        p = hidden_chanels[3]
        p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
        self.res4 = Residual_Block(p1, p2, p3, p4)

        self.res=nn.Sequential(self.res1,self.res2,self.res3,self.res4,nn.MaxPool2d(2))
        self.relu= nn.ReLU(inplace=True)
        self.linear=nn.Linear(hidden_chanels[3][1] * 1 * 1 , out_chanel)
    def forward(self,x):
      out = self.conv1(x)
      out =self.res(out)
      out = F.adaptive_avg_pool2d(out, [1, 1])
      out = out.view(out.size(0), -1)
      out = self.linear(out)
      return out     

resnet_hidden =[[64, 32, 1, 1], [32, 64, 1, 1], [64, 128, 1, 1], [128, 256, 1, 1]]


config = {
    'VGG16': [64, 64, 'N',
              128, 128, 'N',
              256, 256,'N',
              512, 512,'N'],
}


class Vgg_model(nn.Module):
    def __init__(self, vgg_name):
        super(Vgg_model, self).__init__()
        self.features = self.get_layers(config[vgg_name])
        hidden=100352
        self.linear = nn.Linear(hidden, 30)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_layers(self, config):
        layers = []
        in_channels = 3
        for x in config:
            if x != 'N':

                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers=nn.Sequential(*layers)
        return layers


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=5,stride=2,padding=2),    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.conv3=nn.Sequential( nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=384 * 14 * 14, out_features=4096),
            nn.ReLU()
        )
        self.linear2=nn.Sequential(nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=30),)
    def forward(self,x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(-1, 384 * 14 * 14)
            x = self.linear1(x)
            x=self.linear2(x)
            return x
        
        
        
class Attention(nn.Module):
    def __init__(self, in_planes):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Resnet_attention(nn.Module):

    def __init__(self,in_chanel,hidden_chanels,out_chanel):
        super(Resnet_attention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_chanel, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(3, stride=2, padding=1))

        p = hidden_chanels[0]

        p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
        self.res1 = Residual_Block(p1, p2, p3, p4)

        p = hidden_chanels[1]
        p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
        self.res2 = Residual_Block(p1, p2, p3, p4)

        p = hidden_chanels[2]
        p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
        self.res3 = Residual_Block(p1, p2, p3, p4)

        p = hidden_chanels[3]
        p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
        self.res4 = Residual_Block(p1, p2, p3, p4)

        self.res = nn.Sequential(self.res1, self.res2, self.res3, self.res4, nn.MaxPool2d(2))
        self.relu = nn.ReLU(inplace=True)
        self.linear=nn.Linear(hidden_chanels[3][1] * 1 * 1 , out_chanel)
        self.softmax= nn.Softmax(dim=1)
    def forward(self,x):
      out = self.conv1(x)

      out = self.ca(out) * out
      out =self.pool(out)
      out =self.res(out)
      out = self.ca1(out) * out
      out=self.pool1(out)
      out = F.adaptive_avg_pool2d(out, [1, 1])
      out = out.view(out.size(0), -1)
      out = self.linear(out)
      return out


cost = nn.CrossEntropyLoss().to(device)
model1 = ResNet_basic(3, resnet_hidden, 30)
optimizer = torch.optim.Adam(model1.parameters(),lr=1e-4) 
model1 = model1.to(device)
model1, val_acc_history1, train_acc_history1,valid_losses1, train_losses1= model_training(model1, dataloaders, cost, optimizer, device, num_epochs=50)



cost = nn.CrossEntropyLoss().to(device)
model2 = Baseline()
optimizer = torch.optim.Adam(model2.parameters(),lr=1e-4) 
model2 = model2.to(device)
model2, val_acc_history2, train_acc_history2,valid_losses2, train_losses2= model_training(model2, dataloaders, cost, optimizer, device, num_epochs=50)


# In[13]:


cost = nn.CrossEntropyLoss().to(device)
model3=Vgg_model('VGG16')
optimizer = torch.optim.Adam(model3.parameters(),lr=1e-4) 
model3 = model3.to(device)
model3, val_acc_history3, train_acc_history3,valid_losses3, train_losses3= model_training(model3, dataloaders, cost, optimizer, device, num_epochs=50)


# In[18]:


cost = nn.CrossEntropyLoss().to(device)
resnet_hidden =[[64, 32, 1, 1], [32, 64, 1, 1], [64, 128, 1, 1], [128, 256, 1, 1]]
model4=Resnet_attention(3, resnet_hidden, 30)
optimizer = torch.optim.Adam(model4.parameters(),lr=1e-4) 
model4 = model4.to(device)
model4, val_acc_history4, train_acc_history4,valid_losses4, train_losses4= model_training(model4, dataloaders, cost, optimizer, device, num_epochs=50)


# In[21]:


model_name=['Resnet','Base_model','Vgg','Attention_model']
train_loss_list=[train_losses1,train_losses2,train_losses3,train_losses4]
valid_loss_list=[valid_losses1,valid_losses2,valid_losses3,valid_losses4]
acc_train_list=[train_acc_history1,train_acc_history2,train_acc_history3,train_acc_history4]
acc_valid_list=[val_acc_history1,val_acc_history2,val_acc_history3,val_acc_history4]
models=[model1,model2,model3,model4]
for i1,i2 in zip(model_name,train_loss_list):
    plt.plot(range(50),i2,label=i1)
    
plt.legend()
plt.show()


# In[31]:


for i1,i2 in zip(model_name,valid_loss_list):
    plt.plot(range(50),i2,label=i1)

plt.xlabel('Training epochs')
plt.ylabel('Valid loss')
plt.legend()
plt.show()


# In[32]:


for i1,i2 in zip(model_name,acc_train_list):
    i2=[i.data.cpu() for i in i2]
    plt.plot(range(50),i2,label=i1)

plt.xlabel('Training epochs')
plt.ylabel('Train accuracy')
plt.legend()
plt.show()

for i1,i2 in zip(model_name,acc_valid_list):
    i2=[i.data.cpu() for i in i2]
    plt.plot(range(50),i2,label=i1)

plt.xlabel('Training epochs')
plt.ylabel('Valid accuracy')
plt.legend()
plt.show()


# In[26]:


import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',  
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")  

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


metrics={}
for i,model in zip(model_name,models):
    time1=time.time()
    model.eval() 
    y_true = []
    y_pred = []
    probas=[]
    with torch.no_grad():
        for images, labels in dataloaders['valid']:

              images = torch.Tensor(images).float().to(device)
              labels = torch.Tensor(labels).float().to(device)
              outputs = model(images)
              _, predicted = torch.max(outputs.data, 1)
              proba=torch.softmax(outputs,axis=1)
              y_true.extend(labels.cpu().numpy())
              y_pred.extend(predicted.cpu().numpy())
              probas.extend(proba.cpu().numpy())
        time2=time.time()
        time_use=round((time2-time1)/len(y_true),4)
       
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        acc=accuracy_score(y_true, y_pred)
        metrics[i]=[acc,precision,recall,f1,time_use]

    print("Classification Report for", i)
    print(classification_report(y_true, y_pred)) 
    print('\n')


    cnf_matrix = confusion_matrix(y_true, y_pred,normalize='true') 
    plt.figure(figsize=(10, 8))
    np.set_printoptions(precision=2)

    class_names = train_dataset.classes
    plt.figure()
    plot_confusion_matrix(cnf_matrix   
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()

metrics=pd.DataFrame(metrics,columns=metrics.keys()).T
metrics.columns=['Accuracy','Precision','Recall','F1 score','Average cls Time']


