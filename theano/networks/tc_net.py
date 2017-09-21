import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import PIL.Image as Image
from .base_network import BaseNetwork


class Network(BaseNetwork):
    def __init__(self,
                 train_list_raw,
                 test_list_raw,
                 png_folder,
                 batch_size,
                 dropout,
                 l2,
                 mode,
                 batch_norm,
                 **kwargs):

        print("==> not used params in DMN class:", kwargs.keys())
        self.train_list_raw = train_list_raw
        self.test_list_raw = test_list_raw
        self.png_folder = png_folder
        self.batch_size = batch_size
        self.dropout = dropout
        self.l2 = l2
        self.mode = mode
        self.batch_norm = batch_norm

        # self.input_var = T.tensor4('input_var')
        # self.answer_var = T.ivector('answer_var')

        print("==> building network")
        example = np.random.uniform(size=(self.batch_size, 1, 256, 858),
                                    low=0.0,
                                    high=1.0).astype(np.float32)
        answer = np.random.randint(low=0,
                                   high=176,
                                   size=(self.batch_size,))

        # inicia a modelagem sequencial
        model = Sequential()
        # nao tem camada de input

        # CONV-RELU-POOL 1
        model.add(Conv2D(16, (7, 7), strides=1, activation='relu',
                         input_shape=(1, 256, 858)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding=2))
        if (self.batch_norm):
            model.add(BatchNormalization())

        # CONV-RELU-POOL 2
        model.add(Conv2D(32, (5, 5), strides=1, activation='relu',
                         input_shape=(1, 256, 858)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding=2))
        if (self.batch_norm):
            model.add(BatchNormalization())

        # CONV-RELU-POOL 3
        model.add(Conv2D(64, (3, 3), strides=1, activation='relu',
                         input_shape=(1, 256, 858)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding=2))
        if (self.batch_norm):
            model.add(BatchNormalization())

        # CONV-RELU-POOL 4
        model.add(Conv2D(128, (3, 3), strides=1, activation='relu',
                         input_shape=(1, 256, 858)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding=2))
        if (self.batch_norm):
            model.add(BatchNormalization())

            # CONV-RELU-POOL 5
        model.add(Conv2D(128, (3, 3), strides=1, activation='relu',
                         input_shape=(1, 256, 858)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding=2))
        if (self.batch_norm):
            model.add(BatchNormalization())

            # CONV-RELU-POOL 6
        model.add(Conv2D(256, (3, 3), strides=1, activation='relu',
                         input_shape=(1, 256, 858)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 2), padding=2))
        if (self.batch_norm):
            model.add(BatchNormalization())

            # DENSE 1
        network = layers.DenseLayer(incoming=network, num_units=1024,
                                    nonlinearity=rectify)
        if (self.batch_norm):
            network = layers.BatchNormLayer(incoming=network)
        if (self.dropout > 0):
            network = layers.dropout(network, self.dropout)
        print(layers.get_output(network).eval({self.input_var: example}).shape)

        # Last layer: classification
        network = layers.DenseLayer(incoming=network, num_units=3,
                                    nonlinearity=softmax)
        print(layers.get_output(network).eval({self.input_var: example}).shape)

        self.params = layers.get_all_params(network, trainable=True)
        self.prediction = layers.get_output(network)

        print("==> param shapes", [x.eval().shape for x in self.params])

        self.loss_ce = lasagne.objectives.categorical_crossentropy(
            self.prediction, self.answer_var).mean()
        if (self.l2 > 0):
            self.loss_l2 = self.l2 * lasagne.regularization.regularize_network_params(
                network,
                lasagne.regularization.l2)
        else:
            self.loss_l2 = 0
        self.loss = self.loss_ce + self.loss_l2

        # updates = lasagne.updates.adadelta(self.loss, self.params)
        updates = lasagne.updates.momentum(self.loss, self.params,
                                           learning_rate=0.003)

        if self.mode == 'train':
            print("==> compiling train_fn")
            self.train_fn = theano.function(
                inputs=[self.input_var, self.answer_var],
                outputs=[self.prediction, self.loss],
                updates=updates)

        print("==> compiling test_fn")
        self.test_fn = theano.function(
            inputs=[self.input_var, self.answer_var],
            outputs=[self.prediction, self.loss])

    def say_name(self):
        return "tc_net_mod"

    def read_batch(self, data_raw, batch_index):

        start_index = batch_index * self.batch_size
        end_index = start_index + self.batch_size

        data = np.zeros((self.batch_size, 1, 256, 858), dtype=np.float32)
        answers = []

        for i in range(start_index, end_index):
            answers.append(int(data_raw[i].split(',')[1]))
            # print(int(data_raw[i].split(',')[1]))
            name = data_raw[i].split(',')[0]
            # print(data_raw[i].split(',')[0])
            path = self.png_folder + name + ".png"
            # print(path)
            im = Image.open(path)
            data[i - start_index, 0, :, :] = np.array(im).astype(
                np.float32) / 256.0

        answers = np.array(answers, dtype=np.int32)
        return data, answers