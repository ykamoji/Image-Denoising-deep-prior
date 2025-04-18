import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from dip_skip import EncDec
from utils import imread

# Load clean and noisy image
#im = imread('../data/denoising/saturn.png')
#noise1 = imread('../data/denoising/saturn-noisy.png')

file = 'saturn'

im = imread(f'../data/denoising/{file}.png')
noise1 = imread(f'../data/denoising/{file}-noisy.png')

error1 = ((im - noise1)**2).sum()


################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Create network
net = EncDec().to(device)


# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
clean_img = torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).transpose(2,3)

# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size())
eta.detach()

###
# Your training code goes here.
###
net.train()
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 5000
eta, noisy_img, clean_img = eta.to(device), noisy_img.to(device), clean_img.to(device)
train_loss_history = []
test_loss_history = []
best_test_loss = 1000
exp_weight=0.99
out_avg = None
for epoch in range(epochs):
    net.train()
    out = net(eta)

    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    loss = criterion(out, noisy_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss_history.append(loss)

    net.eval()
    with torch.no_grad():
        test_loss = criterion(out, clean_img)
        test_loss_history.append(test_loss.item())

    if test_loss < best_test_loss and test_loss < 100:
        best_test_loss = test_loss

        out_img = out_avg[0, 0, :, :].transpose(0, 1).detach().cpu().numpy()

        plt.figure(3)
        plt.axis('off')

        plt.subplot(131)
        plt.imshow(im, cmap='gray')
        plt.title('Input')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(noise1, cmap='gray')
        plt.title('SE {:.2f}'.format(error1))
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(out_img, cmap='gray')
        plt.title('SE {:.2f}'.format(test_loss))
        plt.axis('off')
        plt.savefig(f"../outputs/task3/{file}/{test_loss.item()}.jpg")
        # plt.show()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]: Loss={train_loss_history[-1]:.5f}  Eval Loss={test_loss_history[-1]:.5f}")
