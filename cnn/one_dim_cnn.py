"""
Description: This example demonstrates the use of Convolution1D for text classification.
Adapted from IMDB text classification, https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
"""
from __future__ import print_function

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.utils.vis_utils import plot_model
from numpy import arange
import joblib


class OneDimCnn(object):
    classes_ = {}

    def __init__(self, max_features=1e5, maxlen=500, batch_size=32, embedding_dims=50, filters=250, kernel_size=3,
                 hidden_dims=250, epochs=10):
        """Constructor"""
        self.max_features = int(max_features)
        self.maxlen = int(maxlen)
        self.batch_size = batch_size
        self.embedding_dims = int(embedding_dims)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.hidden_dims = int(hidden_dims)
        self.epochs = int(epochs)
        print(vars())

    def train(self, df, save=False):
        """Trains the data

        :param df: Dataframe with column labeled, 'input' and 'label'
        :param save: Saves the model and tokenizer.
        :return tokenizer, model"""

        print('Loading data...')

        # load training dataset
        train_df, test_df = self.train_test_split(df)

        # map classes
        self.map_classes(df)

        # create tokenizer
        train_lines = [l for l in train_df['input']]
        test_lines = [l for l in test_df['input']]
        tokenizer = self.create_tokenizer(train_lines)

        # encode data
        x_train = self.encode_text(tokenizer, train_lines, self.maxlen)
        x_test = self.encode_text(tokenizer, test_lines, self.maxlen)
        print('Train shape: ', x_train.shape)
        print('Test shape: ', x_test.shape)

        # encode training classes
        y_train = self.encode_classes(train_df['label'].values)
        # encode test classes
        y_test = self.encode_classes(test_df['label'].values)

        print('Building model...')
        # define model
        model = self.define_model()

        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x_test, y_test))

        if save:
            # save the model
            model.save('multich_cnn_model.h5')
            joblib.dump(tokenizer, 'keras_onedim_tokenizer.pkl')

        return tokenizer, model

    def define_model(self):
        """Defines 1-dimensional convolutional neural network."""

        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen))
        model.add(Dropout(0.2))

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(self.filters, self.kernel_size, padding='valid', activation='relu', strides=1))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        if len(self.classes_) > 2:
            model.add(Dense(len(self.classes_), activation='sigmoid'))

            # use 'categorical_crossentropy' for multi-classification problem
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        if len(self.classes_) == 2:
            # We project onto a single unit output layer, and squash it with a sigmoid:
            model.add(Dense(1, activation='sigmoid'))

            # use 'binary_crossentropy' for binary classes
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        print(model.summary())
        plot_model(model, show_shapes=True, to_file='onedim_cnn.png')

        return model

    @staticmethod
    def train_test_split(df, test_size=0.25, random_state=None):
        """Splits dataframe to train and test

        :param df: A dataframe with column labels, 'input' and 'label'
        :param test_size: fraction of test size
        :param random_state:  int or numpy.random.RandomState, optional
                              Seed for the random number generator (if int), or numpy RandomState object
        :return train, test: dataframe of train and test"""

        test_size = int(df['label'].shape[0] * test_size)
        train_size = df['label'].shape[0] - test_size
        shuffled = df.sample(frac=1, random_state=random_state)
        shuffled = shuffled.reset_index(drop=True)

        train = shuffled.loc[:train_size, :]
        test = shuffled.loc[train_size:, :]

        return train, test

    def map_classes(self, df):
        # map classes to int
        classes = df['label'].unique()
        classes_int = arange(df['label'].unique().shape[0])
        self.classes_ = dict(zip(classes, classes_int))

    def encode_classes(self, arr):
        classes = self.classes_
        for key, val in classes.items():
            arr[arr == key] = val
        return arr

    @staticmethod
    def create_tokenizer(lines):
        """Fits a tokenizer

        :param lines: list of texts to turn into sequences"""
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    @staticmethod
    def encode_text(tokenizer, lines, length):
        """Encodes a list of lines."""
        # integer encode
        encoded = tokenizer.texts_to_sequences(lines)
        # pad encoded sequences
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded
