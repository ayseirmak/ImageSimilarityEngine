import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn as nn
import pytorch_ssim

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import data
import model6
import train_val
import train_val2
import embedding
import config
import ssim


""""SSIM classes"""

class MS_SSIM_Loss(ssim.MS_SSIM):
    def forward(self, img1, img2):
        return ( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )

class SSIM_Loss(ssim.SSIM):
    def forward(self, img1, img2):
        return ( 1 - super(SSIM_Loss, self).forward(img1, img2) )

def train():
    #transforms = T.ToTensor() # Normalize the pixels and convert to tensor.# Normalize the pixels and convert to tensor.
    full_dataset = data.FolderDataset(config.DATASET_PATH, config.TRANSFORMS) # Create folder dataset.
    print(full_dataset.__len__())
    
    train_size = int(config.TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(train_size)
    print(val_size)
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size]) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=1,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE,num_workers=1,pin_memory=True)
    #full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=config.BATCH_SIZE,num_workers=4)
    
    loss_fn = nn.MSELoss() # We use Mean squared loss which computes difference between two images.
    
    
    """ SSIM loss  """ 
    loss_fn2 = SSIM_Loss(data_range=1.0, size_average=True, channel=3)
    
    encoder = model6.ConvEncoder() # Our encoder model
    decoder = model6.ConvDecoder() # Our decoder model
    
    device = "cuda"  # GPU device
    
    # Shift models to GPU
    encoder.to(device)
    decoder.to(device)
    
    # Both the enocder and decoder parameters
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(autoencoder_params, lr=config.LR) # Adam Optimizer
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.0001, last_epoch=-1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001, last_epoch=-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.2,
                                                        patience=2,
                                                        threshold=1e-4,
                                                        min_lr=0.00005)
    lrs = []
    
    # Time to Train !!!
    # Usual Training Loop
    val_losses=[]
    train_losses=[]
    tl=0
    vl=0
    for epoch in tqdm(range(config.EPOCHS)):
            train_loss = train_val.train_step(encoder, decoder, train_loader, loss_fn, optimizer,scheduler,epoch,device=device)
            lrs.append(optimizer.param_groups[0]["lr"])
            train_losses.append(train_loss)
            print(f"Epochs = {epoch}, Training Loss : {train_loss}")
            
            val_loss = train_val.val_step(encoder, decoder, val_loader, loss_fn, device=device)
            val_losses.append(val_loss)
            
            tl+=train_loss
            vl+=val_loss
            
            print(f"Epochs = {epoch}, Validation Loss : {val_loss}")
    
            # Simple Best Model saving
            if val_loss < config.MAX_LOSS:
                config.MAX_LOSS=val_loss
                print("Validation Loss decreased, saving new best model")
                torch.save(encoder.state_dict(), config.ENC_STATE)
                torch.save(decoder.state_dict(), config.DEC_STATE)
                
                # config.tb.add_scalar("Test loss", vl, epoch)  
                # config.tb.add_scalar("Train Loss", tl,epoch) 
    
    plt.plot(lrs)            # config.tb.add_scalar("Train Loss", train_losses,epoch)    
    plt.plot(train_losses,label="train losses")
    plt.plot(val_losses,label="val losses")
    plt.legend()
    plt.show()
    # config.tb.close()

if __name__ == '__main__':
    train()