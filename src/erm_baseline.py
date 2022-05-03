import functools
from random import shuffle
import numpy as np
import torch
import pandas as pd
import decord as de
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import FileLister, FileOpener
import os
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import argparse

sns.set()

def get_basic_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(32*32*3, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 2))

class Net(torch.nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(256, 2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def get_id_from_path(x):
    return Path(x).parts[-2]

def get_label_from_id(df, x):
    return df.loc[x, "character"]

def is_png(x):
    return x.endswith(".png")

def is_group(df, group, x):
    return bool(df.loc[x[0], "group"] == group)

def get_second_element(x):
    return x[1]

def get_train_val_test(dataset_path):
    df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"), index_col="idx")
    dp = FileLister(dataset_path, recursive=True).filter(is_png)
    dp_id = dp.map(get_id_from_path)
    dp = dp_id.zip(dp)

    splits = []
    for i in range(5):
        splits.append(dp.filter(functools.partial(is_group, df, i)))

    def get_dataset(dp):
        dp_id, dp_paths = dp.unzip(2)
        dp_label = dp_id.map(functools.partial(get_label_from_id, df))

        dp_images = FileOpener(dp_paths, "b")
        dp_images = dp_images.routed_decode(imagehandler("torch"))
        dp_images = dp_images.map(get_second_element)

        return dp_images.zip(dp_label)


    splits = [get_dataset(dp) for dp in splits]

    train = list(splits[0])
    for s in splits[1:-1]:
        train += list(s)
    shuffle(train)
    nb_train = int(len(train)*0.8)
    train, val = train[:nb_train], train[nb_train:]

    train = DataLoader(train, shuffle=True, batch_size=1000)
    val = next(iter(DataLoader(val, shuffle=True, batch_size=1000)))
    test = next(iter(DataLoader(list(splits[-1]), shuffle=True, batch_size=1000)))
    return train,val,test

def image_grid(array, ncols=16):
    index, height, width, channels = array.shape
    nrows = index//ncols
    
    img_grid = (array.reshape(nrows, ncols, height, width, channels)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, channels))
    
    return img_grid

def to_PIL_grid(array):
    return Image.fromarray((255*image_grid(array.permute(0,2,3,1).numpy())).astype(np.uint8))

def save_val_test_sample(val, test):
    val_img = (255*image_grid(val[0][:256, ...].permute(0,2,3,1).numpy())).astype(np.uint8)
    test_img = (255*image_grid(test[0][:256, ...].permute(0,2,3,1).numpy())).astype(np.uint8)
    Image.fromarray(val_img).save("figures/val_sample.png")
    Image.fromarray(test_img).save("figures/test_sample.png")

def train_model(train, val, test, epochs=20, nb_seeds=8, lr=0.001, weight_decay=0.0):
    results = []

    for k in tqdm(range(nb_seeds)):
        model = Net().to(0)
        optim = torch.optim.Adam(model.parameters(), lr=lr ,weight_decay=weight_decay)

        val_accs = []
        test_accs = []
        for i in tqdm(range(epochs), leave=False):
            for x,y in tqdm(train, leave=False):
                optim.zero_grad()
                x = x.to(0)
                y = y.to(0)
                y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                optim.step()
        
                val_accs.append((model(val[0].to(0)).argmax(dim=-1)==val[1].to(0)).float().mean().cpu().item())
                test_accs.append((model(test[0].to(0)).argmax(dim=-1)==test[1].to(0)).float().mean().cpu().item())

        results.append(pd.DataFrame({"step":range(len(val_accs)), "acc":val_accs, "type":"val", "model":k}))
        results.append(pd.DataFrame({"step":range(len(test_accs)), "acc":test_accs, "type":"test", "model":k}))
    return pd.concat(results).reset_index(drop=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a model on the dynamicprint dataset given as argument')
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--nb-seeds", type=int, default=8, help="Number of seeds")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")

    args = parser.parse_args()
    train, val, test = get_train_val_test(args.dataset_path)
    save_val_test_sample(val, test)
    results = train_model(train, val, test, args.epochs, args.nb_seeds, args.lr, args.weight_decay)
    results = pd.DataFrame(results)
    sns.lineplot(data=results, x="step", y="acc", hue="type")
    plt.savefig('figures/val_test_accuracy.png')
    plt.show()
