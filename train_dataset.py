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


path_google_drive_main = '/content/drive/MyDrive/'
path_saved_model = '/content/drive/MyDrive/saved_data/model/'
path_saved_weights = '/content/drive/MyDrive/saved_data/weights/'



def get_dataset(csv_file_name, is_one_hot_encoding):
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

    # train_x = [int(x) for x in row for row in train_x]
    # labels = [int(x) for x in labels]
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
    return model
    # model.save('policy_black.h5')
    # model.save('policy_white.h5')


def get_dataset(csv_name,is_one_hot_encoding):
    csv_name = path_google_drive_main+csv_name
    name = csv_name[:-4]  # ~~~.csv에서 .csv자르기
    data_x, data_y = get_dataset(csv_name, is_one_hot_encoding=one_hot_encoding)
    print("데이터 로딩 성공")
    data_x = reshape_to_15_15_1(data_x)
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


def train_model(model,csv_name,is_one_hot_encoding,batch_size):
    name = csv_name[:-4]  # ~~~.csv에서 .csv자르기
    checkpoint_path = name+'.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,mode='auto')
    plateau = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto')
    model.summary()
    data_x,data_y = get_dataset(csv_name,is_one_hot_encoding=is_one_hot_encoding)
    model.fit(data_x,data_y,batch_size=batch_size, epochs=10, shuffle=True, validation_split=0.1,callbacks=[cp_callback,plateau])
    model.save(f'{name}.h5')
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



# import csv
# import numpy as np
#
# import TrainNetwork
# from TrainingPipeline import TrainPipeline
#
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
# def start_train_dataset():
#     print("\n------------------------------------")
#     print("크기 : 15")
#     print("Colab에서만 가능")
#     print("-----------------------------------\n")
#     size = 15
#     train_environment = 1  # 코랩
#     ai_lib = input("라이브러리 이름 : ")
#     if ai_lib == 'tf':
#         ai_lib = 'tensorflow'
#
#     csv_file_name = input("csv파일 이름 : ")
#     data_x, data_y = get_dataset(f'train_data/{csv_file_name}',is_one_hot_encoding=True)
#     train_data_len = len(data_x)
#
#     init_num = int(input("시작 횟수 : "))
#
#     if ai_lib == 'tensorflow':
#         model_file = f'/content/drive/MyDrive/train_by_data/tf_model_{size}_{init_num}_model'
#         tf_lr_data = f'/content/drive/MyDrive/train_by_data/tf_model_{size}_{init_num}.pickle'
#     else:
#         print("지원되지 않는 라이브러리")
#         quit()
#
#     print("\n일반 + 기존 MCTS: 0")
#     print("테스트 + 기존 MCTS : 1")
#     print("일반 + 신규 MCTS : 2")
#     print("테스트 + 신규 MCTS : 3")
#     input_mode = int(input())
#     if input_mode == 0:
#         is_test_mode = False
#         is_new_MCTS = False
#     elif input_mode == 1:
#         is_test_mode = True
#         is_new_MCTS = False
#     elif input_mode == 2:
#         is_test_mode = False
#         is_new_MCTS = True
#     elif input_mode == 3:
#         is_test_mode = True
#         is_new_MCTS = True
#
#     print("훈련 데이터 길이 :",train_data_len)
#
#     model = TrainNetwork.get_model()
#     model.summary()
#     model.fit(data_x,data_y,batch_size=5120, epochs=10, shuffle=True, validation_split=0.1)
#     model.save(f'/content/drive/MyDrive/new_test_model.h5')
#     print("모델 저장 완료")
#     # training_pipeline = TrainPipeline(size, size, train_environment, ai_lib, model_file=model_file,
#     #                                   start_num=init_num, tf_lr_data=tf_lr_data, is_test_mode=is_test_mode,
#     #                                   is_new_MCTS=is_new_MCTS,is_trainset_mode=True)
#     # training_pipeline.run_train_set(train_data_len,data_x,data_y)
#
#
# if __name__ == '__main__':
#     print("데이터셋 로딩 테스트")
#
#
#
