import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import and sort train & test sets

train = pd.read_csv('./data/train/train_data')
train = train.sort_values('img_id', ascending=True)
# test = pd.read_csv('./data/test/test_data')

# Import train & test images

train_img = []
test_img = []

for i in tqdm(range(train.shape[0])):
    img = image.load_img('./data/train/train_img/' + train['img_id'][i].astype('str') + '.jpeg', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_img.append(img)

X = np.array(train_img)
print(X)