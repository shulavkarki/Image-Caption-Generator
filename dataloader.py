import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import config
from models.model import CNNtoRNN
from dataset import TextImageDataset, MyCollate

def get_loader(
    csv_path,
    transform,
    shuffle=True,
    pin_memory=True,
):
    """
    It takes a csv file, transforms the data, and returns a dataloader
    
    :param csv_path: path to the csv file
    :param transform: This is the transform that we will apply to the images
    :param shuffle: whether to shuffle the data, defaults to True (optional)
    :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before
    returning them, defaults to True (optional)
    :return: dataloader is a dictionary with two keys: train and val. Each key has a dataloader object.
    """
    df = pd.read_csv(csv_path).sample(frac=1).reset_index(drop=True)
    #each image has 5 unqiue captions  where 4 of em is used for train and rest for val puprpose
    # trainidx = [i for i in range(0, 40455) if i % 5 != 0]
    # #val
    # validx = [i for i in range(0, 40455, 5)]
    train_split = int(df.shape[0]*0.85)
    #dataset
    dataset = {}
    dataset['train'] = TextImageDataset(df[:train_split], transform=transform)
    dataset['val'] = TextImageDataset(df[train_split:].reset_index(drop=True), transform=transform)
    
    pad_idx = dataset['train'].vocab.stoi["<PAD>"]
    
    dataloader = {x: DataLoader(
        dataset=dataset[x],
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        drop_last=True, 
    ) for x in ['train','val']}

    return dataloader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    
    loader_, dataset = get_loader(
        "data/captions.csv", transform=transform
    )
    # return loader_, dataset
    # from models.model import CNNtoRNN
    for idx, (img, cap) in enumerate(loader_['val']):
        print(img, cap)
        print(img.shape)
        print(cap.shape)
        break
    #     if idx==1:
    #         break
    #     out = CNNtoRNN(40, 5, 40, 5)
    #     output = out(img ,cap)
    #     print("Output: {output}")
    #     print(cap.shape)
    #     print(img.shape)
    #     print(out)

    # # caption = []

    # image = torch.randn(1, 3, 224, 224)
    # from models.model import encoderCNN
    # x = encoderCNN(image)


    # prediction = caption_img(image)
    # print(prediction)

