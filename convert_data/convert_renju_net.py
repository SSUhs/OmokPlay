# https://www.renju.net/game/ 에서 다운 받을 수 있는 데이터
import numpy as np
import codecs
from copy import deepcopy
from random import randint

from tensorflow.python.keras.utils.np_utils import to_categorical

from convert_data.renju_state import State

TRAIN_SIZE = 1000
TRAIN_COUNT = 1
TEST_SIZE = 2000 여기 수정하기

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


for data_num in range(TRAIN_COUNT):
    b_count = 0
    w_count = 0
    b_state = np.zeros(shape=(TRAIN_SIZE, 2, 15, 15))
    w_state = np.zeros(shape=(TRAIN_SIZE, 2, 15, 15))
    b_move = np.zeros(shape=TRAIN_SIZE)
    w_move = np.zeros(shape=TRAIN_SIZE)

    while True:
        line = f.readline()
        if not line:
            break
        if line[:6] != "<move>":
            continue

        i = 6

        sym = randint(0, 7)
        s = State()
        m_count = 0

        while 96 < ord(line[i]) < 112:
            r = ord(line[i]) - 97
            if 47 < ord(line[i + 2]) < 58:
                c = int(line[i + 1:i + 3]) - 1
                i += 4
            else:
                c = int(line[i + 1]) - 1
                i += 3

            r, c = symmetry(sym, r, c)
            m_count += 1

            if s.check_turn() and b_count < TRAIN_SIZE:
                copy_black = deepcopy(s.black)
                copy_white = deepcopy(s.white)
                b_state[b_count] = [copy_black, copy_white]  # [a,b] 형태에서 a가 list, b도 list이므로 이중 리스트
                b_move[b_count] = 15 * r + c
                b_count += 1
            elif not s.check_turn() and w_count < TRAIN_SIZE:
                w_state[w_count] = [deepcopy(s.white), deepcopy(s.black)]
                w_move[w_count] = 15 * r + c
                w_count += 1

            if b_count % 10000 == 0 and b_count < TRAIN_SIZE:
                print(b_count)
            if b_count == TRAIN_SIZE and w_count == TRAIN_SIZE:
                print(b_count)
                break

            s = s.next(15 * r + c)

        if b_count == TRAIN_SIZE and w_count == TRAIN_SIZE:
            break

    b_state = b_state.transpose(0, 2, 3, 1)
    w_state = w_state.transpose(0, 2, 3, 1)
    b_move = to_categorical(b_move, 225)
    w_move = to_categorical(w_move, 225)

    print("데이터", data_num + 1)
    print(b_state.shape, w_state.shape)
    print(b_move.shape, w_move.shape)
    print()