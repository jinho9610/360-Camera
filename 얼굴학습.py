'''
torch와 torchvision 버전을 잘 맞춰줘야함
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import mtcnn
import cv2
import time
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False,
               min_face_size=40)  # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet1 = InceptionResnetV1(pretrained='vggface2').eval()
resnet = torch.load('new_res2.pt').eval()
resnet1 = torch.load('4xxxfinetuned_IRV1.pt').eval()

dataset = datasets.ImageFolder('photos')  # photos folder path
# dataset = datasets.ImageFolder('photos')  # photos folder path
# accessing names of peoples from folder names
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}


def collate_fn(x):
    return x[0]


def train():
    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = []  # list of names corrospoing to cropped photos
    # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
    embedding_list = []

    for img, idx in loader:
        # img.show()
        #Wimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        face, prob = mtcnn0(img, return_prob=True)
        if face is not None and prob > 0.92:
            emb = resnet(face.unsqueeze(0))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

    # save data
    data = [embedding_list, name_list]
    # print(type(data))
    # print(type(data[0]), type(data[1]))
    # print(data[1])
    torch.save(data, 'new_data8.pt')  # saving data.pt file
    print(data[0][0])
    # torch.save(data, 'data.pt')  # saving data.pt file


if __name__ == "__main__":
    train()
