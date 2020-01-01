#!/usr/bin/env python
# coding: utf-8

# ### ダウンロードしたinput.zipをGoogleDriveの直下に置いてから実行してください！

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system("cp '/content/drive/My Drive/input.zip' .")
get_ipython().system("unzip -q 'input.zip'")
get_ipython().system("rm 'input.zip'")


# In[ ]:


get_ipython().system('pip install -U catalyst')
get_ipython().system('pip install efficientnet_pytorch')


# ## Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import os, random
import cv2
import collections
import time
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import pytorch as AT
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback, MixupCallback
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner
from catalyst.contrib.criterion import FocalLossMultiClass
from catalyst.contrib.schedulers import OneCycleLRWithWarmup

from datetime import datetime
from torchsummary import summary
from imblearn.over_sampling import SMOTE

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEED = 255  # FF
set_global_seed(SEED)


# ## Data overview

# In[2]:


path = './year_estimate/'
train = pd.read_csv('./year_estimate/train_labels1979to1980.csv', header=None, names=["id", "label"])
train['label'] = train['label'] - 1980
time_run = datetime.now().strftime('%Y%m%d_%H%M%S')
logdir = "./year_estimate/results/"+ time_run
os.makedirs(logdir, exist_ok=True)

num_class = 39


# In[3]:


train.head()


# In[4]:


n_train = len(os.listdir(f'{path}/train_images'))
n_test = len(os.listdir(f'{path}/test_images'))
print(f'There are {n_train} images in train dataset')
print(f'There are {n_test} images in test dataset')


# In[5]:


train['label'].value_counts().plot(kind="bar")


# ## Dataset

# In[6]:


class FFDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train',
                 transforms=albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]), ):

        self.df = df
        self.data = df['id']
        self.label = df['label']
        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.data[idx]
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img)
        org_img = augmented['image']
        label = self.label[idx]
        
        f = np.fft.fft2(org_img)
        fr = f.real
        fr = torch.from_numpy(fr).float()
        fi = f.imag
        fi = torch.from_numpy(fi).float()
        img = torch.cat((org_img, fr, fi), 0)
        
        return img, label

    def __len__(self):
        return len(self.df)


# In[7]:


img_height = 256
img_width = 256

# train用のデータ拡張
data_transforms = albu.Compose([
    albu.Resize(img_height, img_width),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomBrightnessContrast(p=0.3),
    albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
    albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.10, rotate_limit=45, p=0.5),
    albu.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),   # white noise shouuld be a useful data aug
    albu.Normalize(),
    AT.ToTensor()
])

# test用のデータ拡張
data_transforms_test = albu.Compose([
    albu.Resize(img_height, img_width),
    albu.Normalize(),
    AT.ToTensor()
    ])


# In[8]:


# 学習データを訓練用と検証用に分割


skf = StratifiedKFold(n_splits=10, random_state=SEED)
train_index, test_index = skf.split(train['id'], train['label']).__next__()

train_df = train.iloc[train_index].reset_index(drop=True)
valid_df = train.iloc[test_index].reset_index(drop=True)

#Oversampling using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(train_index.reshape(-1, 1), train_df['label'].to_list()) #Resample and expand the train_index
train_df_res = train.iloc[X_res.reshape(-1,)].reset_index(drop=True)

train_dataset = FFDataset(df=train_df_res, datatype='train', transforms=data_transforms)
valid_dataset = FFDataset(df=valid_df, datatype='valid', transforms=data_transforms_test)

num_workers = 0
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader


# ## Model

# In[127]:


## Self-defined model
'''
class Resnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        for p in self.resnet.parameters():
            p.requires_grad = True
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, n_classes)
         
    def forward(self, x):
        x = self.resnet(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
#        x = self.dropout(x)
        out = F.softmax(self.fc(x))
        return out
'''


# In[128]:


'''
loadmodel = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_class)
firstlayer = Conv2dStaticSamePadding(9, 40, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=[img_height, img_width])

model = nn.Sequential(firstlayer, *list(loadmodel.children())[1:])
'''


# In[129]:


'''

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes, in_chan=3):
        super().__init__()
        self.firstlayer = Conv2dStaticSamePadding(in_chan, 40, kernel_size=(3, 3), stride=(2, 2), bias=False, image_size=[img_height, img_width])
        eNet = EfficientNet('efficientnet-b3', num_classes=num_class)
        self.eNet = nn.Sequential(*list(eNet.children())[1:])
        
        
    def forward(self, x):
        x = self.firstlayer(x)
        x = self.eNet(x)
        return 
'''


# In[9]:


#model = Resnet(num_class)
#model = CustomEfficientNet(num_classes=num_class, in_chan=9)
model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_class, in_channels=9)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, factor=0.33, patience=3, min_lr=1e-6, verbose=True)


# ## Model training

# In[10]:


print(logdir)
# model runner
num_epochs = 80
runner = SupervisedRunner()

#resume_dir=None

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[
               AccuracyCallback(num_classes=num_class, accuracy_args=[1,3]),
               # EarlyStoppingCallback(patience=5, min_delta=0.001),
               CheckpointCallback(save_n_best=num_epochs, resume_dir=None),
               ],
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
    )


# ## Predicting

# In[36]:


submission = pd.read_csv('year_estimate/sample_submission.csv', header=None, names=["id", "label"])

test_dataset = FFDataset(df=submission, datatype='test', transforms=data_transforms_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# In[40]:


model.eval()
predictions = runner.predict_loader(
    model=model,
    loader=test_loader,
    resume=f"{logdir}/checkpoints/53.pth",
    verbose=True
    )

# print(predictions.shape)


# In[41]:


class_names = sorted(list(train['label'].unique()))
pred_list = []
for i in range(n_test):
    probabilities = torch.softmax(torch.from_numpy(predictions[i]), dim=0)
    label = probabilities.argmax().item()
    pred_list.append(class_names[label] + 1980)


# In[42]:


submission['label'] = pred_list
submission.to_csv(f"{logdir}/submission.csv", header=False, index=False)


# In[120]:


summary(model, (9, 256, 256))


# In[123]:


print(model)


# In[ ]:




