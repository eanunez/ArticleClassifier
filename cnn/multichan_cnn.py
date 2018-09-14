from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
import pandas as pd
from numpy import arange


class MultiChanCnn(object):
    """
    Name: Multi-Channel Convolutional Neural Network
    Description:
     A class of N-Gram Multichannel Convolutional Neural Network which takes in Pandas' dataframe as an input data.
     This is an adaptation of Jason Brownlee which can be found at,

     https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/

     This approach was first described by Yoon Kim in his 2014 paper titled
     “Convolutional Neural Networks for Sentence Classification.”

     "A multi-channel convolutional neural network for document classification
     involves using multiple versions of the standard model with different sized kernels.
     This allows the document to be processed at different resolutions or different
     n-grams (groups of words) at a time, whilst the model learns how to best integrate
     these interpretations." - Jason Brownlee, Ph.D
     """

    classes_ = {}

    def __init__(self):
        """Constructor"""

    def train(self, df, save=False):
        """Trains the data

        :param df: Dataframe with column labeled, 'input' and 'label'
        :param save: Saves the model.
        :return returns the trained model."""

        # load training dataset
        train_df, test_df = self.train_test_split(df)

        # create tokenizer
        train_lines = [l for l in train_df['input']]
        tokenizer = self.create_tokenizer(train_lines)

        # calculate max document length
        length = self.max_length(train_lines)

        # calculate vocabulary size
        vocab_size = len(tokenizer.word_index) + 1

        print('Max document length: %d' % length)
        print('Vocabulary size: %d' % vocab_size)

        # encode data
        train_x = self.encode_text(tokenizer, train_lines, length)
        print('Train shape: ', train_x.shape)

        # define model
        model = self.define_model(length, vocab_size)

        # map classes
        self.map_classes(df)

        # encode training classes
        train_y = self.encode_classes(train_df['label'].values)

        # fit model
        model.fit([train_x, train_x, train_x], train_y, epochs=10, batch_size=16)

        # evaluate model
        print('=== Evaluating Model ===')
        test_lines = [l for l in test_df['input']]
        test_x = self.encode_text(tokenizer, test_lines, length)
        print('Test shape: ', test_x.shape)

        # encode test classes
        test_y = self.encode_classes(test_df['label'].values)

        # evaluate model on test dataset
        loss, acc = model.evaluate([test_x, test_x, test_x], test_y, verbose=0)
        print('Test Accuracy: %f' % (acc * 100))

        if save:
            # save the model
            model.save('multich_ccn_model.h5')

        return model

    @staticmethod
    def train_test_split(df, test_size=0.25, random_state=None):
        """Splits dataframe to train and test

        :param df: A dataframe with column labels, 'input' and 'label'
        :param test_size: fraction of test size
        :param random_state:  int or numpy.random.RandomState, optional
                              Seed for the random number generator (if int), or numpy RandomState object
        :return train, test: dataframe of train and test"""

        train, test = pd.DataFrame([]), pd.DataFrame([])
        label = df['label'].unique()
        test_size = int(df['label'].shape[0] * test_size)
        train_size = df['label'].shape[0] - test_size
        shuffled = df.sample(frac=1, random_state=random_state)
        shuffled = shuffled.reset_index(drop=True)
        for lbl in label:
            train = shuffled[shuffled.loc[:, 'label'] == lbl].loc[:train_size, :]
            test = shuffled[shuffled.loc[:, 'label'] == lbl].loc[train_size:, :]

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
    def max_length(lines):
        """Calculates the maximum document length."""
        return max([len(s.split()) for s in lines])

    @staticmethod
    def encode_text(tokenizer, lines, length):
        """Encodes a list of lines."""
        # integer encode
        encoded = tokenizer.texts_to_sequences(lines)
        # pad encoded sequences
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded

    @staticmethod
    def define_model(length, vocab_size):
        """Defines the multi-channel CNN model"""
        # channel 1
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, 100)(inputs1)
        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # channel 2
        inputs2 = Input(shape=(length,))
        embedding2 = Embedding(vocab_size, 100)(inputs2)
        conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # channel 3
        inputs3 = Input(shape=(length,))
        embedding3 = Embedding(vocab_size, 100)(inputs3)
        conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(10, activation='relu')(merged)
        outputs = Dense(1, activation='sigmoid')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # compile
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # summarize
        print(model.summary())
        # plot_model(model, show_shapes=True, to_file='multichannel.png')
        return model
