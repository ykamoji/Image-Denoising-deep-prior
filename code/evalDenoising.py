import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from dip import EncDec
from utils import imread
from utils import gaussian
from scipy import ndimage
from scipy import signal

# Load clean and noisy image
#im = imread('../data/denoising/saturn.png')
#noise1 = imread('../data/denoising/saturn-noisy.png')

file = 'lena'

im = imread(f'../data/denoising/{file}.png')
noise1 = imread(f'../data/denoising/{file}-noisy.png')

error1 = ((im - noise1)**2).sum()

print('Noisy image SE: {:.2f}'.format(error1))

plt.figure(1)

# Original image
plt.subplot(141)
plt.imshow(im, cmap='gray')
plt.title('Input')

# Noisy version
plt.subplot(142)
plt.imshow(noise1, cmap='gray')
plt.title('Noisy image SE {:.2f}'.format(error1))

# Apply Gaussian filter
plt.subplot(143)
gaussian_filter = gaussian(7, 2)
im_gauss = ndimage.convolve(noise1, gaussian_filter, mode="nearest")
error_gauss = ((im - im_gauss)**2).sum()
plt.imshow(im_gauss, cmap='gray')
plt.title('Gaussian SE {:.2f}'.format(error_gauss))

# Apply Median filter
plt.subplot(144)
im_med = signal.medfilt(noise1, 7)
error_med = ((im - im_med)**2).sum()
plt.imshow(im_med, cmap='gray')
plt.title('Median SE {:.2f}'.format(error_med))
plt.show(block=True)


################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Create network
net = EncDec().to(device)


# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
clean_img = torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).transpose(2,3)

fig, ax = plt.subplots(1, 5, figsize=(14, 7))
with torch.no_grad():
    for i in range(5):
        noise_input = torch.randn(*noisy_img.size())
        noise_input = noise_input.to(device)
        out = net(noise_input)
        out_img = out[0, 0, :, :].transpose(0, 1).detach().cpu().numpy()
        ax[i].imshow(out_img)
        ax[i].set_title(f"Random {i+1}")
        ax[i].axis('off')
    # plt.savefig(f"../outputs/task3/random_images")
    plt.show()

# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size())
eta.detach()

###
# Your training code goes here.
###
net.train()
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 500
eta, noisy_img, clean_img = eta.to(device), noisy_img.to(device), clean_img.to(device)
train_loss_history = []
test_loss_history = []
for epoch in range(epochs):
    net.train()
    out = net(eta)

    loss = criterion(out, noisy_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss_history.append(loss)

    net.eval()
    with torch.no_grad():
        test_loss = criterion(out, clean_img)
        test_loss_history.append(test_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]: Loss={train_loss_history[-1]:.5f}  Eval Loss={test_loss_history[-1]:.5f}")

plt.plot(np.arange(1, len(train_loss_history) + 1), [loss.item() for loss in  train_loss_history],
         'r-', label="Training loss history")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
# plt.savefig(f"../outputs/task3/{file}_train_loss_{epochs}")
plt.show()

plt.plot(np.arange(1, len(test_loss_history) + 1), test_loss_history, 'b-', label="Testing loss history")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
# plt.savefig(f"../outputs/task3/{file}_test_loss_{epochs}")
plt.show()

net.eval()
with torch.no_grad():
    out = net(eta)
    out_img = out[0, 0, :, :].transpose(0,1).detach().cpu().numpy()

error1 = ((im - noise1)**2).sum()
error2 = ((im - out_img)**2).sum()

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
plt.title('SE {:.2f}'.format(error2))
plt.axis('off')
# plt.savefig(f"../outputs/task3/{file}")
plt.show()
