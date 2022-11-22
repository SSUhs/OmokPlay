import copy

import numpy as np
import codecs
import csv
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from random import randint
from tensorflow.keras.models import load_model
import keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from keras.models import Model
import pickle

path_google_drive_main = '/content/drive/MyDrive/'
path_saved_model = '/content/drive/MyDrive/saved_data/model/'
path_saved_weights = '/content/drive/MyDrive/saved_data/weights/'


def convert_load_dataset(csv_file_name, is_one_hot_encoding,type_train):
    data_x_p_black = []  # 흑 정책망 input
    data_x_p_white = []  # 백 정책망 input
    labels_p_black = []  # 흑 정책망 레이블
    labels_p_white = []  # 백 정책망 레이블
    data_x_v = []  # 가치망 input
    labels_v = []  # 가치망 레이블
    data_y_p_black = None
    data_y_p_white = None
    data_y_v = None

    if type_train >= 3:
        print(f"존재 하지 않는 type_train : {type_train}")
        quit()

    print("\n데이터 셋 로딩 시작..")
    with open(csv_file_name, 'r') as f:
        next(f, None)
        reader = csv.reader(f)
        count_read = 0
        skip_count = 0
        # 헤더 : move 위치 / black_value / 누가 돌을 놓을차례(흑1, 백2) / 상태~
        for row in reader: # row는 문자열 리스트
            count_read += 1
            if float(row[1]) <= -100: # 승부 판별 불가능
                skip_count+=1
                continue
            if type_train == 0: # 흑 정책망 학습
                if int(float(row[1]) == 1) and int(float(row[2]) == 1):  # 흑이 이기는 경우이면서 흑이 돌을 놓을 차례인 경우
                    labels_p_black.append(int(float(row[0])))
                    data_x_p_black.append(row[3:])
            elif type_train == 1: # 백 정책망 학습
                if int(float(row[1]) == 2) and int(float(row[2]) == 2):  # 백이 이기는 경우이면서 백이 돌을 놓을 차례인 경우
                    labels_p_white.append(int(float(row[0])))
                    data_x_p_white.append(row[3:])
            elif type_train == 2:
                data_x_v.append(row[3:])
                labels_v.append(row[1])
            if count_read % 4000 == 0:
                print("현재까지 읽은 row 수 :",count_read)

    if len(data_x_p_black) >=1:
        data_x_p_black = np.array(data_x_p_black, dtype=np.float32)
    if len(data_x_p_white) >= 1:
        data_x_p_white = np.array(data_x_p_white, dtype=np.float32)
    if len(labels_p_black) >= 1:
        labels_p_black = np.array(labels_p_black, dtype=np.int32)
        data_y_p_black = labels_p_black
        data_y_p_black = data_y_p_black.astype(dtype=np.int32)
    if len(labels_p_white) >= 1:
        labels_p_white = np.array(labels_p_white, dtype=np.int32)
        data_y_p_white = labels_p_white
        data_y_p_white = data_y_p_white.astype(dtype=np.int32)
    if len(labels_v) >= 1:
        labels_v = np.array(labels_v, dtype=np.float32)
        data_y_v = labels_v
        data_y_v = data_y_v.astype(dtype=np.float32)


    if is_one_hot_encoding:
        print("0 1만으로 표현하지 않으므로 사용 X")
        quit()
        # a = np.array(labels_p_black)
        # b = np.zeros((len(labels_p_black), 225))
        # b[np.arange(len(labels_p_black)), a] = 1
        # data_y_p_black = b
        #
        # a = np.array(labels_p_white)
        # b = np.zeros((len(labels_p_white), 225))
        # b[np.arange(len(labels_p_white)), a] = 1
        # data_y_p_white = b
        #
        # a = np.array(labels_v)
        # b = np.zeros((len(labels_v), 225))
        # b[np.arange(len(labels_v)), a] = 1
        # data_y_v = b

    return data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v

def reshape_to_15_15_1(data):
    return K.reshape(data,[-1,15,15,1])

# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_model(model_type):
    model = None
    if model_type == 0: # 정책망 + Conv2D 6개
        model = Sequential()
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(15, 15, 1)))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(225, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0025), metrics=['acc'])
    elif model_type == 1:
        model = Sequential()
        model.add(Conv2D(64, 7, activation='relu', padding='same', input_shape=(15, 15, 1)))
        model.add(Conv2D(128, 7, activation='relu', padding='same'))
        model.add(Conv2D(256, 7, activation='relu', padding='same'))
        model.add(Conv2D(128, 7, activation='relu', padding='same'))
        model.add(Conv2D(64, 7, activation='relu', padding='same'))
        model.add(Conv2D(1, 1, activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(225, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0025), metrics=['acc'])
    elif model_type == 2: # 정책 + 가치망
        model = get_not_sequential_model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0025), metrics=['acc'])
    elif model_type == 3: # 가치망
        model = Sequential()
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(15, 15, 1)))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(225, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.00003), metrics=[rmse])
    return model
    # model.save('policy_black.h5')
    # model.save('policy_white.h5')

def get_not_sequential_model():
    board_size = 15
    in_x = network = tf.keras.Input((board_size, board_size,1))
    l2_const = 1e-4  # coef of l2 penalty
    # code20221121183414
    network = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                                     activation="relu", kernel_regularizer=l2(l2_const))(network)
    network = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                                     activation="relu", kernel_regularizer=l2(l2_const))(network)
    network = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same",
                                     activation="relu", kernel_regularizer=l2(l2_const))(network)
    # action policy layers
    policy_net = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1),
                                        activation="relu",
                                        kernel_regularizer=l2(l2_const))(network)
    policy_net = tf.keras.layers.Flatten()(policy_net)
    policy_net = tf.keras.layers.Dense(board_size * board_size, activation="softmax", kernel_regularizer=l2(l2_const))(policy_net)
    # state value layers
    value_net = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), activation="relu",
                                       kernel_regularizer=l2(l2_const))(network)
    value_net = tf.keras.layers.Flatten()(value_net)
    value_net = tf.keras.layers.Dense(64, kernel_regularizer=l2(l2_const))(value_net)
    value_net = tf.keras.layers.Dense(1, activation="tanh", kernel_regularizer=l2(l2_const))(value_net)
    # model_ = Model(in_x, [policy_net, value_net]) # 가치망 출력은 제거 (학습 데이터에 가치망 label이 존재하지 않음. 따라서 가중치 연산만 미리 해두고 나중에 parameter만 불러와서 연산)
    model_ = Model(in_x, [policy_net,value_net])
    return model_

# pv_type : 'seperate' >> policy, value 분리망
def get_dataset(csv_name,is_one_hot_encoding,pv_type,type_train):
    csv_name = path_google_drive_main+csv_name
    if pv_type == 'seperate':
        data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v= convert_load_dataset(csv_name, is_one_hot_encoding=is_one_hot_encoding,type_train=type_train)
    else:
        print("미구현")
    print("데이터 로딩 성공")
    if type_train == 0:
        data_x_p_black = reshape_to_15_15_1(data_x_p_black)
    elif type_train == 1:
        data_x_p_white = reshape_to_15_15_1(data_x_p_white)
    elif type_train == 2:
        data_x_v = reshape_to_15_15_1(data_x_v)
    else:
        print("존재하지 않는 type")
        quit()

    # 주의!! sequential이 아닌 방식의 경우, data_y가 [a,b]형태가 되어야함
    return data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v

def make_new_model():
    while True:
        model_type = int(input("모델 타입 선택 : "))
        model = get_model(model_type)
        if model is None:
            print("존재하지 않는 모델입니다\n")
            continue
        else:
            break
    return model

# 이건 전체 모델 파일을 불러오는 방식
def load_saved_model(model_file):
    model_file = path_saved_model+ model_file
    model = tf.keras.models.load_model(model_file)
    return model


# 가중치만 로딩
# 따라서 모델 구조가 동일하면 다른 모델끼리도 가중치만 로딩해서 사용할 수 있음
def load_saved_weights(model_instance, weights_file):
    weights_file = path_saved_weights+weights_file
    model_instance.load_weights(weights_file)
    return model_instance

def save_pickle(save_path,model):
    net_params = model.get_weights()
    pickle.dump(net_params, open(save_path, 'wb'), protocol=2)


def train_model(model_policy_b,model_policy_w,model_value,csv_name,is_one_hot_encoding,batch_size):
    name = csv_name[:-4]  # ~~~.csv에서 .csv자르기
    checkpoint_path = name+'.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,mode='auto')
    plateau = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto')
    print("\n------------------정책망 (흑) ------------------")
    model_policy_b.summary() # 어차피 흑이나 백이나 망은 동일한 구조이므로 하나만 출력
    print("\n------------------가치망------------------")
    model_value.summary()

    type_train = int(input("훈련할 대상 (메모리 부족으로 따로 해야함) : 0(흑 정책망) / 1(백 정책망) / 2(가치망)"))
    data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v= get_dataset(csv_name,is_one_hot_encoding=is_one_hot_encoding,pv_type='seperate',type_train=type_train)


    # 여기서 오류 나면, 데이터가 없는 것이나 부족한 경우. 예를 들어 백이 승리한 데이터가 없는 경우
    # 또는 정답지 데이터가
    if type_train == 0:
        data_y_p_black = to_categorical(data_y_p_black)
        print("\n------------------Shape------------------")
        print(f'data_x_p_black : {data_x_p_black.shape}')
        print(f'data_y_p_black : {data_y_p_black.shape}')
        print("\n------------------흑 정책망 훈련을 시작합니다------------------")
        model_policy_b.fit(data_x_p_black, data_y_p_black, batch_size=batch_size, epochs=10, shuffle=True,
                           validation_split=0.1, callbacks=[cp_callback, plateau])
        model_policy_b.save_weights(f'{path_google_drive_main + name}_black_weights')  # 확장자는 일단 pickle이긴 한데 정확 X
        model_policy_b.save(f'{path_google_drive_main + name}_black.h5')
        save_pickle(f'{path_google_drive_main + name}_black.pickle', model_policy_b)
    elif type_train == 1:
        print("\n------------------Shape------------------")
        data_y_p_white = to_categorical(data_y_p_white)
        print(f'data_y_p_white : {data_y_p_white.shape}')
        print("\n------------------백 정책망 훈련을 시작합니다------------------")
        model_policy_w.fit(data_y_p_black, data_y_p_white, batch_size=batch_size, epochs=10, shuffle=True,
                           validation_split=0.1, callbacks=[cp_callback, plateau])
        model_policy_w.save_weights(f'{path_google_drive_main + name}_white_weights')  # 확장자는 일단 pickle이긴 한데 정확 X
        model_policy_w.save(f'{path_google_drive_main + name}_white.h5')
        save_pickle(f'{path_google_drive_main + name}_white.pickle', model_policy_w)
    elif type_train == 2:
        data_y_v = to_categorical(data_y_v)
        print("\n------------------가치망(흑의 승 기준) 훈련을 시작합니다------------------")
        model_value.fit(data_x_v, data_y_v, batch_size=batch_size, epochs=10, shuffle=True, validation_split=0.1,
                        callbacks=[cp_callback, plateau])
        model_value.save_weights(f'{path_google_drive_main + name}_value_weights')  # 확장자는 일단 pickle이긴 한데 정확 X
        model_value.save(f'{path_google_drive_main + name}_value.h5')
        save_pickle(f'{path_google_drive_main + name}_value.pickle', model_value)
    else:
        print("없는 경우 - type-train")

    # print("\n------------------Shape------------------")
    # print(f'data_x_p_black : {data_x_p_black.shape}')
    # print(f'data_x_p_white : {data_x_p_white.shape}')
    # print(f'data_x_v : {data_x_v.shape}')
    # print(f'data_y_p_black : {data_y_p_black.shape}')
    # print(f'data_y_p_white : {data_y_p_white.shape}')
    # print(f'data_y_v : {data_y_v.shape}')
    # print(f"배치 사이즈 : {batch_size}")

    print("모델 최종 저장이 완료되었습니다")


# 훈련된 모델을 테스트 data set을 이용해서 테스트
def test_model(model,csv_file_name,one_hot_encoding):
  print("\n-----------------실제 테스트-----------------\n")
  csv_file_name = path_google_drive_main + csv_file_name
  print("데이터 로딩을 시작합니다")
  data_x,data_y = get_dataset(csv_file_name,is_one_hot_encoding=one_hot_encoding)
  print("데이터 로딩 성공")
  print("데이터 수 :",len(data_x))
  data_x = reshape_to_15_15_1(data_x)
  print("테스트 시작")
  test_loss, test_acc = model.evaluate(data_x,data_y,verbose=2)
  print("Test Accuracy :", test_acc)
  print("Test Loss : ", test_loss)


if __name__ == '__main__':
    to_do = int(input("처음 부터 생성 : 0 / 이어서 학습 : 1 /테스트는 2"))
    csv_file = input(f'사용할 csv 파일 : )')
    one_hot_encoding = False
    batch_size = None
    if to_do == 0:
      print("\n정책망 선택")
      model_policy_b = make_new_model()
      model_policy_w = copy.deepcopy(model_policy_b)
      print("\n가치망 선택")
      model_value = make_new_model()
    elif to_do == 1:
      model_file_name = input(f"이어서 학습할 모델 파일 (기본 경로 : {path_saved_model}")
      model = load_saved_model(model_file_name)
    elif to_do == 2:
      model_file_name = input(f"테스트에 사용할 모델 파일 (기본 경로 : {path_saved_model}")
      model = load_saved_model(model_file_name)
    else:
      print("없는 경우")
      quit()

    if to_do == 0 or to_do == 1:
        train_model(model_policy_b,model_policy_w,model_value,csv_file,is_one_hot_encoding=one_hot_encoding,batch_size=512)
    elif to_do == 2:
        test_model(model,csv_file_name=csv_file,one_hot_encoding=one_hot_encoding)


