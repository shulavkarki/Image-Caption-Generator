import torch
import config
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as f

from dataloader import get_loader
from models.model import CNNtoRNN

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

loader_, dataset = get_loader(
    "data/processed.csv", transform=transform
)


# torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Running on {device}.")
# Hyperparameters
vocab_size = len(dataset['train'].vocab)
num_layers = 2
learning_rate = 3e-4
max_length = 25

# initialize model, loss etc 
model = CNNtoRNN(config.EMBED_SIZE, config.HIDDEN_SIZE, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train():
    train_losses = []
    val_losses = []
    for epoch in range(config.EPOCH):
        print(f"Epoch: {epoch+1}")
        for mode in ['train', 'val']:
            running_loss = 0.0
            for idx, (imgs, captions) in tqdm(
                enumerate(loader_[mode]), total=len(loader_[mode]), leave=False
            ):
                imgs = imgs.to(device)
                captions = captions.to(device)

                outputs = model(imgs, captions)
                # print(outputs.shape)
                label_one_hot = f.one_hot(captions, vocab_size)
                loss = criterion(
                    outputs.contiguous().view(-1, vocab_size), label_one_hot.view(config.BATCH_SIZE*max_length, -1).float(), 
                )
                optimizer.zero_grad()
                if mode=='train':
                    loss.backward()
                    optimizer.step()
                # print(loss.item())
                running_loss+=loss.item()
                
            if mode=='train':
                train_losses.append(running_loss)
                print(f'Training Loss: {running_loss:.3f}')
            else:
                val_losses.append(running_loss)
                print(f'Validation Loss: {running_loss:.3f}')
                print('---------------------------------------------')
            if (epoch+1) % 5== 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch+1,
                    # 'weight': weight,
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                }
                torch.save(checkpoint, f'{config.MODEL_SAVE_PATH}checkpoint{epoch+1}.pth')
if __name__ == "__main__":
    train()
