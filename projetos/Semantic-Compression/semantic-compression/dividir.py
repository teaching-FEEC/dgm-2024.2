import os
import shutil
import random


path='dataset/images'
train_path = 'data/train/'
val_path = 'data/val'

images = os.listdir(f'{path}')
random.shuffle(images)
for image in images[8000:]:
    shutil.copy(f'{path}/{image}',
        f'{val_path}/{image}')
for image in images[:8000]:
    shutil.copy(f'{path}/{image}',
        f'{train_path}/{image}')
