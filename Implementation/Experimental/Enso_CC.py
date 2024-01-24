####################~~~ENSO~~~######################
#unsupervised framework for microscopy images analysis
#Developed by Dr J. Griffie
#Scientific project: BALTIC funded by an MSCA fellowship
#If used, please cite associated publication
#####################################################################

#Workflow
#Enso takes as input microscopy images (x_train and x_test) 
#and generate a latent space reprensentation which it saves (Latent.txt) 
#as well as the loss function (loss.txt), the model (model.pt) and a visual
#containing a subset of images and their corresponding synthetic pair
######################################################################

#[1] Import libraries/packages
import numpy as np
import matplotlib.pyplot as plot
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ##servers
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

x_train = np.zeros((4926, 1, 64, 192), dtype=np.float32)
for i in range(1, 2464):
    path = "Data sets\\CC\\"
    tmp = np.loadtxt(path + str(i) + ".csv", dtype=np.float32, delimiter=",")
    x_train[i-1] = tmp[:64, :192]
    x_train[i-1] = x_train[i-1]/x_train[i-1].max()

for i in range(2464, 4927):
    path = "Data sets\\CC_mir\\"
    tmp = np.loadtxt(path + str(i-2463) + ".csv", dtype=np.float32, delimiter=",")
    x_train[i-1] = tmp[:64, :192]
    x_train[i-1] = x_train[i-1]/x_train[i-1].max()

x_test = np.zeros((2463, 1, 64, 192), dtype=np.float32)#x_test consists of the 
# images for which the coordinates in the latent space will be infered and saved
# i.e., it consists of your data sets without any data augmentation
for i in range(1, 2464):
    path = "Data sets\\CC\\"
    tmp = np.loadtxt(path + str(i) + ".csv", dtype=np.float32, delimiter=",")
    x_test[i-1] = tmp[:64, :192]
    x_test[i-1] = x_test[i-1]/x_test[i-1].max()

x_train=torch.from_numpy(x_train)
x_test=torch.from_numpy(x_test)

#[3]VAE architecture
class VAE(nn.Module):
    #zDim stands for the dimension of the latent space. Typically we vary it from 1 to 3
    def __init__(self, imgChannels=1, featureDim=(64/16)*(192/16)*256, zDim=2):
        #zdim=latent space
        super(VAE, self).__init__()
        self.encConv1 = nn.Conv2d(imgChannels, 4, 3, stride=1, padding='same')
        self.encMax1=nn.MaxPool2d(2, stride=2)
        self.encConv2 = nn.Conv2d(4, 16, 3, stride=1,padding='same')
        self.encMax2=nn.MaxPool2d(2, stride=2)
        self.encConv3 = nn.Conv2d(16, 64, 3, stride=1,padding='same')
        self.encMax3=nn.MaxPool2d(2, stride=2)
        self.encConv4 = nn.Conv2d(64, 256, 3, stride=1,padding='same')
        self.encMax4=nn.MaxPool2d(2, stride=2)
        
        self.encFC0 = nn.Linear(int(featureDim), 256)
        self.encFC00 = nn.Linear(256, 7)
        self.encFC1 = nn.Linear(7, zDim)#mean
        self.encFC2 = nn.Linear(7, zDim)#variance

        self.decFC00 = nn.Linear(zDim, 7)
        self.decFC0 = nn.Linear(7, 256)
        self.decFC1 = nn.Linear(256, int(featureDim))
        
        self.decUp1=nn.Upsample(scale_factor=2)
        self.decConv1 = nn.Conv2d(256, 64, 3, stride=1,padding='same')
        self.decUp2=nn.Upsample(scale_factor=2)
        self.decConv2 = nn.Conv2d(64, 16, 3, stride=1,padding='same')
        self.decUp3=nn.Upsample(scale_factor=2)
        self.decConv3 = nn.Conv2d(16, 4, 3, stride=1,padding='same')
        self.decUp4=nn.Upsample(scale_factor=2)
        self.decConv4 = nn.Conv2d(4, imgChannels, 3, stride=1,padding='same')

    def encoder(self, x):
        x = F.relu(self.encConv1(x))
        x = self.encMax1(x)
        x = F.relu(self.encConv2(x))
        x = self.encMax2(x)
        x = F.relu(self.encConv3(x))
        x = self.encMax3(x)
        x = F.relu(self.encConv4(x))
        x = self.encMax4(x)
        
        x = x.view(-1, int((64/16)*(192/16)*256))
        
        x = self.encFC0(x)
        x = self.encFC00(x)
        
        mu = self.encFC1(x) 
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = F.relu(self.decFC00(z))
        x = F.relu(self.decFC0(x))
        x = F.relu(self.decFC1(x))
        
        x = x.view(-1, 256, int(64/16), int(192/16))
        
        x = self.decUp1(x)
        x = F.relu(self.decConv1(x))
        x = self.decUp2(x)
        x = F.relu(self.decConv2(x))
        x = self.decUp3(x)
        x = F.relu(self.decConv3(x))
        x = self.decUp4(x)
        x = torch.sigmoid(self.decConv4(x))
        return x

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


#[4]RUN
batch_size = 64
learning_rate = 1e-4
num_epochs = 50

net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

path = "Results\\"

for epoch in range(num_epochs):
    print(epoch)
    for batch in range(1,int(1+4926/batch_size)):
        imgs=x_train[(batch-1)*batch_size:batch*batch_size]
        imgs = imgs.to(device)
        
        out, mu, logVar = net(imgs)
        ms_ssim_loss = 1 - ms_ssim(imgs, out, win_size=3, data_range=1, size_average=True )
        
        optimizer.zero_grad()
        ms_ssim_loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, ms_ssim_loss))
    f=open(path+"Loss.txt","a")
    f.write('Epoch {}: Loss {}'.format(epoch, ms_ssim_loss))
    f.close()
    

net.eval()
out_test, mu_test, logVar_test = net(x_test)
out_test=out_test.cpu().detach().numpy()
print(out_test.shape)
mu_test=mu_test.cpu().detach().numpy()
print(mu_test.shape)

path = "Results\\"
np.savetxt(path + "Latent.txt", mu_test, delimiter=",")

torch.save(net, path + 'model.pt') 

n=10
plot.figure(figsize=(15,4))
for k in range(n):
    ax=plot.subplot(2,n,k+1)
    plot.imshow(x_test[k].reshape(64,192))
    ax=plot.subplot(2,n,k+1+n)
    plot.imshow(out_test[k].reshape(64,192))
plot.savefig(path +"Visual.png")
plot.show()

 

