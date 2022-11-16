import csv
import numpy as np


# 학습용 데이터 / 테스트 데이터 둘다 사용 가능
def get_dataset(csv_file_name, is_one_hot_encoding):
    data_x = []
    labels = []

    with open(csv_file_name, 'r') as f:
        next(f, None)
        reader = csv.reader(f)
        for row in reader:
            data_x.append(row[1:])
            labels.append(row[0])

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


def start_train_dataset():
    ai_lib = input("라이브러리 이름 : ")
    if ai_lib == 'tf':
        ai_lib = 'tensorflow'

    csv_file_name = input("csv파일 이름 : ")
    data_x, data_y = get_dataset(f'train_data/{csv_file_name}',is_one_hot_encoding=True)
    train_data_len = len(data_x)

    asdf


    print("훈련 데이터 길이 :",train_data_len)


