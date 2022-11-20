# csv 형태로 변경
# https://gomocup.org/results/ 데이터 셋 변경용

import copy
import csv
from datetime import time, datetime

import numpy as np
import os


def convert_to_label(x, y):
    if x == 0 or y == 0:
        print("시작이 0인 데이터가 존재합니다")
        quit()
    return 15 * (y - 1) + (x - 1)


def convert(folder_name):
    read_folder = folder_name
    current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    output_csv_name_b = f'csv_data/black/output_{current_time}_b.csv'
    output_csv_name_w = f'csv_data/white/output_{current_time}_w.csv'

    file_list = []
    for (root, directories, files) in os.walk(read_folder):
        # for d in directories:
        #     d_path = os.path.join(root, d)
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
            # print(file_path)

    if os.path.isfile(output_csv_name_b) or os.path.isfile(output_csv_name_w):
        print(f'{output_csv_name_b} 또는 {output_csv_name_w} 은 이미 존재하는 csv 데이터입니다')
        quit()
    f_csv_b = open(output_csv_name_b, 'w', encoding='utf-8', newline='')
    f_csv_w = open(output_csv_name_w, 'w', encoding='utf-8', newline='')

    b_states_list = []
    w_states_list = []
    b_labels_list = []
    w_labels_list = []
    all_states = np.zeros([15, 15])

    count_not_dataset_file = 0  # 확장자가 dataset 파일이 아닌 경우 (rec, psq 파일이 아닌 경우)
    count_not_1515 = 0  # 15x15가 아니거나 시작 좌표가 0인 경우
    file_list_len = len(file_list)

    for count in range(len(file_list)):
        data_file_name = file_list[count]
        if not (data_file_name.endswith('.REC') or data_file_name.endswith('.rec') or data_file_name.endswith('.psq')):
            print(f'{data_file_name} 파일은 rec, psq 파일이 아닙니다')
            count_not_dataset_file += 1
            continue

        f = open(data_file_name, 'r')
        list_all = []  # 좌표 말고도 일단 다른 정보도 포함시킨 list
        list = []  # 실제 좌표가 포함된 라인
        while True:
            line = f.readline()
            if not line: break
            list_all.append(line)

        skip_this = False
        for i in range(len(list_all)):  # 혹시 15가 넘어가는 파일이 있으면 스킵
            split = list_all[i].split(",")
            if not (2 <= len(split) <= 3):  #  n,n,n 형태 또는 n,n형태가 아니면 스킵
                continue
            try:
                x = int(split[0])
                y = int(split[1])
                # 혹시 판의 크기가 15를 넘어가는 경우에는 아예 사용하지 않는 파일
                if x > 15 or y > 15 or x == 0 or y == 0:
                    print(f'{data_file_name} 파일의 판 크기에서 발견된 데이터 : ({x},{y}) (0으로 시작하거나 15를 넘는 데이터)')
                    count_not_1515 += 1
                    skip_this = True
                    break
                else:  # 정상적인 좌표 데이터인 경우
                    list.append(list_all[i])
            except:  # split이 안된다면 continue
                continue

        if skip_this:
            continue

        for i in range(len(list)):
            split = list[i].split(",")
            x = int(split[0])
            y = int(split[1])

            if i % 2 == 0:  # 흑
                t = copy.deepcopy(all_states)
                b_states_list.append(copy.deepcopy(t))
                t[x - 1][y - 1] = 1
                label = convert_to_label(x, y)
                all_states = t
                b_labels_list.append(label)
            else:  # 백
                t = copy.deepcopy(all_states)
                w_states_list.append(copy.deepcopy(t))
                t[x - 1][y - 1] = 2
                label = convert_to_label(x, y)
                all_states = t
                w_labels_list.append(label)
        if len(w_states_list) + len(b_states_list) != len(w_labels_list) + len(b_labels_list):
            print("크기가 같지 않습니다")
            quit()
        elif len(w_states_list) != len(w_labels_list):
            print("크기가 같지 않습니다")
            quit()

    f_csv_writer_b = csv.writer(f_csv_b)
    f_csv_writer_w = csv.writer(f_csv_w)
    for i in range(len(b_states_list)):
        output = np.insert(b_states_list[i], 0, int(b_labels_list[i]))
        f_csv_writer_b.writerow(output)

    for i in range(len(w_states_list)):
        output = np.insert(w_states_list[i], 0, int(w_labels_list[i]))
        f_csv_writer_w.writerow(output)

    print("\n---------------변환 성공---------------\n")
    print(f"전체 파일 수 : {file_list_len}")
    print(f'스킵한 전체 파일 수 : {count_not_dataset_file+count_not_1515}')
    print(f"데이터셋 확장자가 아닌 파일 수 : {count_not_dataset_file}")
    print(f"다른 형식 데이터셋 파일 :  {count_not_1515}")
    print(f"흑 전체 상태 수 :  {len(b_states_list)}")
    print(f"백 전체 상태 수 :  {len(w_states_list)}")
    f_csv_b.close()
    f_csv_w.close()


if __name__ == '__main__':
    argstr = input("folder name : ")
    args = argstr.split(" ")
    convert(args[0])
