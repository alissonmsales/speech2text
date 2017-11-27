from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
from random import shuffle
import math
import os
import PIL.Image as Image


def get_data():
    base_label = []
    base_data = []
    a = {}

    # Path to save spectograms
    path = '/media/alissonsales/Files/UTFPR/Pesquisa/pti/locutores/spec/'
    spec = os.listdir(path + 'data_input')
    saida = os.listdir(path + 'label_output')

    # Shuffling to lose order of speakers
    shuffle(spec)

    # Input of the spectograms: output layer
    for s in saida:
        nome = s.split(".")[0]
        im = Image.open(path + 'label_output/' + s)
        im = np.array(im).astype(np.float32) / 255
        a[nome] = im.flatten()

    # Input of the spectograms: input layer
    for s in spec:
        nome = s.split('-')[0]
        base_label.append(a[nome])
        im = Image.open(path + 'data_input/' + s)
        im = np.array(im).astype(np.float32) / 255
        base_data.append(im)

    base_label = np.array(base_label, dtype=np.float32)
    base_data = np.array(base_data, dtype=np.float32)
    base_data = base_data.reshape(len(spec), 216, 13, 1)

    return base_data, base_label


def create_model():
    input_img = Input(shape=(216, 13, 1))

    # Conv Layer 1
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((4, 4), padding='same')(x)

    # Conv Layer 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((4, 4), padding='same')(x)
    x = Flatten()(encoded)

    # Dense Layer 1
    x = Dense(500, activation='relu')(x)

    # Dense Layer 2
    # Embedding output
    x = Dense(80, activation='relu')(x)

    # Dense Layer 3
    decoded = Dense(533, activation='sigmoid')(x)

    model = Model(input_img, decoded)
    model.summary()
    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    return model


if __name__ == '__main__':
    # get data
    data, label = get_data()

    # split data into train and test
    size = math.ceil(len(data)*0.8)
    x_train_data = data[:size]
    x_train_label = label[:size]
    x_test_data = data[size:]
    x_test_label = label[size:]

    # training
    model = create_model()
    model.fit(x_train_data,
              x_train_label,
              epochs=3,
              batch_size=128,
              shuffle=True,
              validation_data=(x_test_data, x_test_label),
              callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
              verbose=1)

    # Save model at the embedding output layer (Dense 3)
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(
                                         'dense_3').output)
    intermediate_layer_model.save('emb_model.h5')
