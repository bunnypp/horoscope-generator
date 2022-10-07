#!/usr/bin/env python
# coding: utf-8

# # Horoscope Generator

# Train a machine learning model and generate horoscopes based on dataset from horoscope.com

# I followed this guide: https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms

# This generator notebook will create models for each of the zodiac signs.

# First import modules:

import tensorflow as tf

# keras module for building LSTM
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import utils as ku
from keras import models as KM
from keras import backend as KBE

# set seeds for reproducability
from numpy.random import seed
tf.random.set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os
import random

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# # Define functions for sequences, creating the model and generating text

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words, tokenizer

def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+ output_word
    return seed_text.title()

# # Generating text with the saved models


def generate_horoscope(sign, starter_text, length):
    signNames = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

    KBE.clear_session()

    curr_sign = sign

    df = pd.read_json('dataset2/'+ curr_sign +'.json')
    corpus = list(df["text"])

    # generate sequences
    inp_sequences, total_words, tokenizer = get_sequence_of_tokens(corpus)

    # obtain predictors and label
    predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences, total_words)

    # load models
    model = KM.load_model("Models/model2-" + curr_sign + ".h5")

    generated_text = generate_text(starter_text, length, model, max_sequence_len, tokenizer)

    return generated_text

def get_input(sign):
    inputs = ["Today", "You", sign, "Consider", "Your", "The", "When", "Be",
            "A", "It", "This", "I"]
    return random.sample(inputs, 1)[0]
