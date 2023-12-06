import torch

def train_step(encoder, decoder, train_loader, loss_fn, optimizer,scheduler,epoch,device):
   
    encoder.train()
    decoder.train()
    iters = len(train_loader)
    for batch_idx, (train_img, target_img,idx) in enumerate(train_loader):
        
        train_img = train_img.to(device)
        target_img = target_img.to(device)
        
        optimizer.zero_grad()

        enc_output = encoder(train_img)
        dec_output = decoder(enc_output)
        
        loss = loss_fn(dec_output, target_img)
        
        loss.backward()

        optimizer.step()
        scheduler.step(epoch + batch_idx / iters)
        #scheduler.step()
    return loss.item()

def val_step(encoder, decoder, val_loader, loss_fn, device):
   
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        for batch_idx, (train_img, target_img,idx) in enumerate(val_loader):
            
            train_img = train_img.to(device)
            target_img = target_img.to(device)

           
            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            loss = loss_fn(dec_output, target_img)
    
    return loss.item()  