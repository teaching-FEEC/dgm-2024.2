import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  
from tensorflow.keras.saving import load_model
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--path_data', type=str, default='01280256032031000010000100_losses.txt')
parser.add_argument('--path_model', type=str, default='model.keras')
parser.add_argument('--path_image', type=str, default='teste.png')

args = parser.parse_args()



file_path = args.path_data
data = np.loadtxt(file_path)



img1_path = args.path_image
img1 = mpimg.imread(img1_path)
x = np.array(img1)

model = load_model(args.path_model)

xhat, what = model(x[np.newaxis])




fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 4, width_ratios=[2, 2, 2, 2]) 

ax1 = fig.add_subplot(gs[0, 0:2]) 
ax1.set_ylim(-1, 2)
ax1.plot(data[0], label='Loss_train')
ax1.plot(data[2], label='Loss_val')
ax1.set_title('D_losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Valor')
ax1.legend()

ax2 = fig.add_subplot(gs[1, 0:2])  
ax2.set_ylim(0, 12)
ax2.plot(data[1], label='Loss_train')
ax2.plot(data[3], label='Loss_val')
ax2.set_title('G_losses')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Valor')
ax2.legend()

ax_img1 = fig.add_subplot(gs[0, 2:4]) 
ax_img1.imshow(img1)
ax_img1.axis('off')  

ax_img2 = fig.add_subplot(gs[1, 2:4])  
ax_img2.imshow(xhat[0])
ax_img2.axis('off') 

plt.tight_layout()
plt.show()
