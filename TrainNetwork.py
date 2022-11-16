import numpy as np
import codecs
from copy import deepcopy
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from random import randint
import keras.backend as K


def reshape_to_15_15_1(data):
    return K.reshape(data,[-1,15,15,1])


def get_model():
    model = Sequential()
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(15, 15, 2)))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(225, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0025), metrics=['acc'])
    return model
    # model.save('policy_black.h5')
    # model.save('policy_white.h5')


if __name__ == '__main__':
    print("모델 생성 테스트")
    model = get_model()
    model.summary()