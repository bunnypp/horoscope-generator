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
import gc
import pickle

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# # Define function for generating text

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

        gc.collect()
        KBE.clear_session()

    return seed_text.title()

# # Generating text with the saved models


def generate_horoscope(sign, starter_text, length):
    signNames = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
             "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

    KBE.clear_session()

    #get tokenizer data
    with open("tokenizer_data/" + sign + "-tokenizerData.pkl", 'rb') as infile:
        data = pickle.load(infile)
        max_sequence_len = data['max_sequence_len']
        tokenizer = data['tokenizer']

    # load models
    model = KM.load_model("Models/model2-" + sign + ".h5")

    generated_text = generate_text(starter_text, length, model, max_sequence_len, tokenizer)

    return generated_text

def get_input(sign):
    inputs = ["Today", "You", sign, "Consider", "Your", "The", "When", "Be",
            "A", "It", "This", "There"]
    return random.sample(inputs, 1)[0]
