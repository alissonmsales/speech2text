import PIL.Image as Image
import numpy as np
from keras import backend as K

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization import  BatchNormalization
from keras import optimizers
from keras.utils import to_categorical


# print(K.image_data_format())
batch_norm = True
dropout = 0.5

train_listfile = open(
        "/media/alissonsales/Files/base_dados/train.csv", "r")
test_listfile = open(
        "/media/alissonsales/Files/base_dados/test.csv", "r")

train_list_raw = train_listfile.readlines()
test_list_raw = test_listfile.readlines()

print("==> %d training examples" % len(train_list_raw))
print("==> %d validation examples" % len(test_list_raw))

train_listfile.close()
test_listfile.close()

png_folder = "/media/alissonsales/Files/base_dados/spec_all/"

train_data = []
train_label = []

test_data = []
test_label = []

# for t in train_list_raw:
for i in range(1000):
    train_label.append(int(train_list_raw[i].split(',')[1]))
    name = train_list_raw[i].split(',')[0]
    path = png_folder + name + ".png"
    im = Image.open(path)
    train_data.append(np.array(im).astype(np.float32) / 256.0)

# for t in test_list_raw:
for j in range(100):
    test_label.append(int(test_list_raw[j].split(',')[1]))
    name = test_list_raw[j].split(',')[0]
    path = png_folder + name + ".png"
    im = Image.open(path)
    test_data.append(np.array(im).astype(np.float32) / 256.0)

train_data = np.array(train_data, dtype=np.float32)
train_data = train_data.reshape(1000, 858, 256, 1)

test_data = np.array(test_data, dtype=np.float32)
test_data = test_data.reshape(100, 858, 256, 1)

# convert class vectors to binary class matrices
train_label = to_categorical(train_label, 3)
test_label = to_categorical(test_label, 3)

""" Construcao da rede """
print(train_data.shape)
# inicia a modelagem sequencial
model = Sequential()
# nao tem camada de input

# CONV-RELU-POOL 1
model.add(Conv2D(16, (7, 7), strides=1, activation='relu',
                 input_shape=(858, 256, 1)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

if (batch_norm):
    model.add(BatchNormalization())

# CONV-RELU-POOL 2
model.add(Conv2D(32, (5, 5), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
#model.add(MaxPooling2D(pool_size=(3, 3)))
if (batch_norm):
    model.add(BatchNormalization())

# CONV-RELU-POOL 3
model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
if (batch_norm):
    model.add(BatchNormalization())

# CONV-RELU-POOL 4
model.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
if (batch_norm):
    model.add(BatchNormalization())

# CONV-RELU-POOL 5
model.add(Conv2D(128, (3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
if (batch_norm):
    model.add(BatchNormalization())
model.summary()
# CONV-RELU-POOL 6
model.add(Conv2D(256, (3, 3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
if (batch_norm):
    model.add(BatchNormalization())

# DENSE 1
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
if (batch_norm):
    model.add(BatchNormalization())
if (dropout > 0):
    model.add(Dropout(dropout))

# Last layer: classification
model.add(Dense(3, activation='softmax'))

# self.params = layers.get_all_params(network, trainable=True)
# self.prediction = layers.get_output(network)
sgd = optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_data, train_label, batch_size=32, epochs=100, verbose=1, validation_data=(test_data, test_label))

score = model.evaluate(test_data, test_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])