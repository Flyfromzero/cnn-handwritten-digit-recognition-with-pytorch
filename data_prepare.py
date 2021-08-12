import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, filename, image_dir, repeat=1):
        '''
        :param filename: TXT file：content: imge_name.jpg label1_id labe2_id...
        :param image_dir: the dir of images
        :param repeat: the times of data being used, default 1
        '''
        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
 
    def __getitem__(self, i):
        """
        get the images and labels
        """
        index = i % self.len
        image_name, label = self.image_label_list[index]
        label = torch.tensor(label)
        image_path = os.path.join(self.image_dir, image_name)
        img = self.data_process(image_path, normalization=True)
        return img, label
 
    def __len__(self):
        """
        get the length of data
        """
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len
 
    def read_file(self, filename):
        """
        read the txt file and get the labels of images
        """
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list
 
    def data_process(self, path, normalization):
        '''
        load and preprocess the images
        '''
        img = imread(path)
        if normalization is True:
            img = self.transform(img)
        img = np.asarray(img,dtype='float64')
        return img

train_filename="./dataset/train.txt"
test_filename="./dataset/test.txt"
train_image_dir='./dataset/train'
test_image_dir='./dataset/test'

batch_size=4
train_data_nums=10

train_data = MyDataset(filename=train_filename, image_dir=train_image_dir,repeat=1)
test_data = MyDataset(filename=test_filename, image_dir=test_image_dir,repeat=1)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=True)

#check the size of data
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_targets = example_targets.t()
example_targets = example_targets.squeeze()
print(example_targets.shape)
print(example_data.shape)
