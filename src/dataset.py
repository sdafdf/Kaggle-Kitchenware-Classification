import torch
import os, glob
import random
import csv
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



class Mydataset(Dataset):
    def __init__(self,root,mode,resize):
        super(Mydataset,self).__init__()
        self.root = root
        self.resize = resize
        self.dirs = os.listdir(self.root)

        self.imgpath,self.csv_path = self.root+self.dirs[0],self.root+self.dirs[-1]
        self.ImgPathList,self.LabelPathList = self.read_csv(self.csv_path,self.imgpath)
        train_img,test_img,train_lab,test_lab = train_test_split(self.ImgPathList,self.LabelPathList,test_size=0.3,random_state=20)
        print(len(train_img))
        if mode=="train":
            self.img_path = train_img
            self.label_path = train_lab
        else:
            self.img_path = test_img
            self.label_path = test_lab
        
    def read_csv(self,csvpath,imgpath):
        self.csv_path = csvpath
        self.imgpath = imgpath
        train = pd.read_csv(csvpath)
        train['image'] = train["Id"].map(lambda x:f"{imgpath}/{x:0>4}.jpg")
        assert len(train["image"])==len(train["label"])
        return list(train["image"]),list(train["label"])

    def __getitem__(self, index):
        img,label = self.img_path[index],self.label_path[index]
        img = Image.open(img).convert("RGB")
        transform = transforms.Compose([transforms.Resize((int(self.resize), int(self.resize))),
                                     transforms.ToTensor(),  # 先变成tensor类型数据，然后在进行下面的标准化
                                     ])

        image = transform(img)
        label_dict = {"glass":0,"cup":1,"spoon":2,"plate":3,"knife":4,"fork":5}
        label = label_dict[label]
        label = torch.tensor(label) 
        return image, label

    def __len__(self):
        return len(self.img_path)


def create_dataloader(data_path, mode, size, batch_size,
                    num_workers=0):  # 用一个函数加载上诉的数据，data_path、mode和size分别是以上定义的Dataset_self(）中的参数，batch_size是一次性输出多少张图像，num_worker是同时处理几张图像
    
    dataset = Mydataset(data_path, mode, size)
    dataloader = DataLoader(dataset, batch_size, num_workers)  # 使用pytorch中的dataloader函数得到数据

    return dataloader


if __name__=="__main__":
    data_path = "F:\\PythonProject\\deep_code\\Kitchenware Classification\\dataset\\"
    dataloader = create_dataloader(data_path,mode="train",size=224,batch_size=64)
    for i,(img,label) in enumerate(dataloader):
        # print(i)
        print(img.shape)
        print(label.shape)
        