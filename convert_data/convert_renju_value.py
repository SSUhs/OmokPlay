import numpy as np
import codecs
from copy import deepcopy
from random import randint
from keras import metrics

from renju_state import State

import keras.backend as K

TRAIN_SIZE = 55
TRAIN_COUNT = 1
TEST_SIZE = 2000


# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


f = codecs.open("renjunet_test.rif", encoding='mac_roman')


def symmetry(i, r, c):
    if i == 0:
        return r, c
    elif i == 1:
        return r, 14 - c
    elif i == 2:
        return c, r
    elif i == 3:
        return c, 14 - r
    elif i == 4:
        return 14 - r, c
    elif i == 5:
        return 14 - r, 14 - c
    elif i == 6:
        return 14 - c, r
    elif i == 7:
        return 14 - c, 14 - r


def sym_list(sym, list):
    new_list = []
    for _ in range(15):
        new_list.append([0] * 15)

    for i in range(15):
        for j in range(15):
            if list[i][j] == 1:
                r, c = symmetry(sym, i, j)
                new_list[r][c] = 1

    return new_list


for data_num in range(TRAIN_COUNT):
    b_state = np.zeros(shape=(TRAIN_SIZE, 2, 15, 15))
    w_state = np.zeros(shape=(TRAIN_SIZE, 2, 15, 15))
    b_move = np.zeros(shape=TRAIN_SIZE)
    w_move = np.zeros(shape=TRAIN_SIZE)
    count = 0

    while True:
        line = f.readline()
        if not line:
            break
        if line[:8] != "<game id":
            continue

        index = 75
        while True:
            if line[index:index + 3] == 'bre':
                break
            else:
                index += 1

        if line[index + 10] == '"':
            result = float(line[index + 9])
            if result == 0: result = -1
        else:
            result = 0
        line = f.readline()

        i = 6

        s = State()
        sym = randint(0, 7)

        b_temp_state = []
        b_temp_move = []
        w_temp_state = []
        w_temp_move = []

        while 96 < ord(line[i]) < 112:
            r = ord(line[i]) - 97
            if 47 < ord(line[i + 2]) < 58:
                c = int(line[i + 1:i + 3]) - 1
                i += 4
            else:
                c = int(line[i + 1]) - 1
                i += 3

            r, c = symmetry(sym, r, c)

            if s.check_turn():
                b_temp_state.append([deepcopy(s.black), deepcopy(s.white)])
                b_temp_move.append(result)

            else:
                w_temp_state.append([deepcopy(s.white), deepcopy(s.black)])
                w_temp_move.append(-result)

            s = s.next(15 * r + c)

        if len(b_temp_move) > 8 and len(w_temp_move) > 8:
            for sym in range(8):
                index = randint(0, len(b_temp_move) - 1)
                temp0 = sym_list(sym, b_temp_state[index][0])
                temp1 = sym_list(sym, b_temp_state[index][1])

                b_state[count] = [temp0, temp1]
                b_move[count] = b_temp_move[index]

                index = randint(0, len(w_temp_move) - 1)
                temp0 = sym_list(sym, w_temp_state[index][0])
                temp1 = sym_list(sym, w_temp_state[index][1])

                w_state[count] = [temp0, temp1]
                w_move[count] = w_temp_move[index]

                count += 1

                if count % 10000 == 0:
                    print(count)
                if count == TRAIN_SIZE:
                    break

        if count == TRAIN_SIZE:
            break


    b_state = b_state.transpose(0, 2, 3, 1) # 예를 들어 원래 a,b,c,d면 a,c,d,b로 바뀌는 것
    w_state = w_state.transpose(0, 2, 3, 1)

    print("데이터", data_num + 1)
    print(b_state.shape, w_state.shape)
    print(b_move.shape, w_move.shape)
    tmp = np.array([b_move]).transpose(1,0)
    print(tmp.shape)
    print(type(b_move[0]))
    print()

    # 모델 학습, 저장
    # model = load_model('value_black.h5')
    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003), metrics=[rmse])
    # history = model.fit(b_state, b_move, batch_size=5120, epochs=20, shuffle=True, validation_split=0.1)
    # model.save('value_black_t.h5')
    #
    # K.clear_session()
    # del model
    # b_state = np.array([0])
    # b_move = np.array([0])
    #
    # model = load_model('value_white.h5')
    # model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003), metrics=[rmse])
    # history2 = model.fit(w_state, w_move, batch_size=5120, epochs=20, shuffle=True, validation_split=0.1)
    # model.save('value_white_t.h5')
    #
    # K.clear_session()
    # del model
    #
    # print()
    # print()

    # 메모리 해제
    w_state = np.array([0])
    w_move = np.array([0])

f.close()