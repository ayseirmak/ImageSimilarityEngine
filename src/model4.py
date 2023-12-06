import torch
import torch.nn as nn

class ConvEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2),stride=2)

        self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2),stride=2)

        self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2),stride=2)

        self.conv4 = nn.Conv2d(256, 512, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2),stride=2)

        self.conv5 = nn.Conv2d(512, 512, (3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2),stride=2)
        
        self.conv6 = nn.Conv2d(512, 1024, (3, 3), padding=(1, 1))
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool6 = nn.MaxPool2d((2, 2),stride=2)

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        
        
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool6(x)
        
        return x
    
class ConvDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.deconv1 = nn.ConvTranspose2d(1024, 512, (3, 3), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(512, 512, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        self.deconv4 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu4 = nn.ReLU(inplace=True)

        self.deconv5 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu5 = nn.ReLU(inplace=True)

        self.deconv6 = nn.ConvTranspose2d(64, 3, (2, 2), stride=(2, 2))
        # self.upsamp1 = nn.UpsamplingBilinear2d(2)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x = self.deconv1(x)
        x = self.relu1(x)
        # print(x.shape)

        x = self.deconv2(x)
        x = self.relu2(x)
        # print(x.shape)

        x = self.deconv3(x)
        x = self.relu3(x)
        # print(x.shape)

        x = self.deconv4(x)
        x = self.relu4(x)
        # print(x.shape)

        x = self.deconv5(x)
        x = self.relu5(x)
        # print(x.shape)
        
        x = self.deconv6(x)
        x = self.relu6(x)
        # print(x.shape)
        return x

#Deneme
# img_random = torch.randn(1, 3, 224, 224)
# img_random2 = torch.randn(1, 3, 224, 224)
# print(img_random.shape)

# enc = ConvEncoder()
# dec = ConvDecoder()

# enc_out = enc(img_random)
# enc_out2 = enc(img_random2)
# print(enc_out.shape)
# print(enc_out2.shape)

# emb = torch.cat((enc_out, enc_out2), 0)
# print(emb.detach().numpy().shape)
# print(emb.shape)


# dec_out = dec(enc_out)
# print(dec_out.shape)
# from torchsummary import summary
# enc = ConvEncoder()
# dec = ConvDecoder()
# print(enc)
# print(dec)
# summary(enc,input_size=(3,224,224))
# summary(dec,input_size=(1024,3,3))

# img_random = torch.randn(1, 3, 224, 224)
# img_random2 = torch.randn(1, 3, 224, 224)










