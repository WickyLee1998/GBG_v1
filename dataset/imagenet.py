import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np

class ImageNetLT(Dataset):

    def __init__(self, root, txt, transform=None, train=True, class_balance=False, ordered_data = False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.num_classes = 1000
        self.train = train
        self.class_balance = class_balance
        self.ordered_data = ordered_data
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))


        self.class_data = [[] for i in range(self.num_classes)]
        self.class_img_path = [[] for i in range(self.num_classes)]
        self.class_labels = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)
            self.class_img_path[y].append(self.img_path[i])
            self.class_labels[y].append(y)
        self.ordered_labels = []
        self.ordered_img_path = []
        self.classes_samples_pointers = {i: [] for i in range(self.num_classes)}
        start = 0
        for i in range(len(self.class_img_path)):
            self.ordered_labels.extend(self.class_labels[i])
            self.ordered_img_path.extend(self.class_img_path[i])
            end = len(self.ordered_labels) - 1
            self.classes_samples_pointers[i] = [start, end]
            start = end + 1
        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
            label = random.randint(0, self.num_classes - 1)
            index = random.choice(self.class_data[label])
            path = self.img_path[index]
        elif self.ordered_data:
            path = self.ordered_img_path[index]
            label = self.ordered_labels[index]
        else:
            path = self.img_path[index]
            label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            if self.train:
                sample1 = self.transform[0](sample)
                sample2 = self.transform[1](sample)
                sample3 = self.transform[2](sample)
                return [sample1, sample2, sample3], label  # , index
            else:
                return self.transform(sample), label



