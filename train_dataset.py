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


def convert_load_dataset(csv_file_name, is_one_hot_encoding):
    data_x = []
    labels = []

    print("\n데이터 셋 로딩 시작..")
    with open(csv_file_name, 'r') as f:
        next(f, None)
        reader = csv.reader(f)
        count_read = 0
        for row in reader:
            data_x.append(row[1:])
            labels.append(int(float(row[0])))  # float은 int타입으로
            count_read += 1
            if count_read % 4000 == 0:
                print("현재까지 읽은 row 수 :",count_read)

    data_x = np.array(data_x, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if is_one_hot_encoding:
        a = np.array(labels)
        b = np.zeros((len(labels), 225))
        b[np.arange(len(labels)), a] = 1
        data_y = b
    else:
        data_y = labels

    data_y = data_y.astype(dtype=np.float32)
    return data_x, data_y

def reshape_to_15_15_1(data):
    return K.reshape(data,[-1,15,15,1])


def get_model(model_type):
    model = None
    if model_type == 0:
        model = Sequential()
        model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(15, 15, 1)))
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
    elif model_type == 2:
        model = get_not_sequential_model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0025), metrics=['acc'])
    return model
    # model.save('policy_black.h5')
    # model.save('policy_white.h5')

def get_not_sequential_model():
    board_size = 15
    in_x = network = tf.keras.layers.InputLayer((board_size, board_size,1))
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
    model_ = Model(in_x, policy_net)
    return model_

def get_dataset(csv_name,is_one_hot_encoding):
    csv_name = path_google_drive_main+csv_name
    # name = csv_name[:-4]  # ~~~.csv에서 .csv자르기
    data_x, data_y = convert_load_dataset(csv_name, is_one_hot_encoding=is_one_hot_encoding)
    print("데이터 로딩 성공")
    data_x = reshape_to_15_15_1(data_x)
    # 주의!! sequential이 아닌 방식의 경우, data_y가 [a,b]형태가 되어야함
    return data_x,data_y

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


def train_model(model,csv_name,is_one_hot_encoding,batch_size):
    name = csv_name[:-4]  # ~~~.csv에서 .csv자르기
    checkpoint_path = name+'.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,mode='auto')
    plateau = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto')
    model.summary()
    data_x,data_y = get_dataset(csv_name,is_one_hot_encoding=is_one_hot_encoding)
    model.fit(data_x,data_y,batch_size=batch_size, epochs=10, shuffle=True, validation_split=0.1,callbacks=[cp_callback,plateau])
    model.save_weights(f'{path_google_drive_main+name}_weights')  # 확장자는 일단 pickle이긴 한데 정확 X
    model.save(f'{path_google_drive_main+name}.h5')
    save_pickle(f'{path_google_drive_main+name}.pickle',model)
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
    one_hot_encoding = True
    batch_size = None
    if to_do == 0:
      model = make_new_model()
    elif to_do == 1:
      model_file_name = input(f"이어서 학습할 모델 파일 (기본 경로 : {path_saved_model}")
      model = load_saved_model(model_file_name)
    elif to_do == 2:
      model_file_name = input(f"테스트에 사용할 모델 파일 (기본 경로 : {path_saved_model}")
      model = load_saved_model(model_file_name)
    else:
      print("없는 경우")
      quit()

    csv_file = input(f'사용할 csv 파일 : )')
    if to_do == 0 or to_do == 1:
        batch_size = int(input("배치 사이즈 : "))
        train_model(model,csv_file,is_one_hot_encoding=one_hot_encoding,batch_size=batch_size)
    elif to_do == 2:
        test_model(model,csv_file_name=csv_file,one_hot_encoding=one_hot_encoding)



# # 학습용 데이터 / 테스트 데이터 둘다 사용 가능
# def get_dataset(csv_file_name, is_one_hot_encoding):
#     data_x = []
#     labels = []
#
#     with open(csv_file_name, 'r') as f:
#         next(f, None)
#         reader = csv.reader(f)
#         for row in reader:
#             data_x.append(row[1:])
#             labels.append(row[0])
#
#     # train_x = [int(x) for x in row for row in train_x]
#     # labels = [int(x) for x in labels]
#     data_x = np.array(data_x, dtype=np.float32)
#     labels = np.array(labels, dtype=np.int32)
#
#     if is_one_hot_encoding:
#         a = np.array(labels)
#         b = np.zeros((len(labels), 225))
#         b[np.arange(len(labels)), a] = 1
#         data_y = b
#     else:
#         data_y = labels
#
#     data_y = data_y.astype(dtype=np.float32)
#     return data_x, data_y
#
#