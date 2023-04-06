import os
import torch
import pickle
from PIL import Image
import tensorflow as tf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence  # pad batch
from vocabulary import Vocabulary
import config
class TextImageDataset(Dataset):
    def __init__(self, df, transform=None, freq_threshold=2):
        self.transform = transform
        self.images = df['image']
        self.captions = df['caption']

        #initlaize Vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocabs = self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = self.images[idx]

        image = Image.open(config.IMAGE_PATH+ '/' + str(img_id)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocabs["<start>"]]
        numericalized_caption.extend(self.vocab.numericalize(caption))
        numericalized_caption.append(self.vocabs["<end>"])

        return image, torch.Tensor(numericalized_caption).long()




class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # print(f"Batch: {batch}")
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = tf.keras.utils.pad_sequences(targets, maxlen=25, value=self.pad_idx, padding='post')
        # print(targets.shape)
        # targets = pad_sequence([t[:50] for t in targets], batch_first=True, padding_value=0)        # targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        # print(type(targets))
        # print(targets.shape)
        targets = torch.from_numpy(targets).unsqueeze(1)
        # print(targets.shape)
        return imgs, torch.Tensor(targets).long()

    