# csv 형태로 변경
import copy
import csv

import numpy as np
import os

def convert_to_label(x,y):
    if x == 0 or y == 0:
        print("시작이 0인 데이터가 존재합니다")
        quit()
    return 15*(y-1)+(x-1)

def convert(folder_name, skip_count):
    read_folder = folder_name
    output_csv_name_b = f'csv_data/output_{folder_name}_b.csv'
    output_csv_name_w = f'csv_data/output_{folder_name}_w.csv'

    file_list = os.listdir(read_folder)
    if os.path.isfile(output_csv_name_b) or os.path.isfile(output_csv_name_w):
        print(f'{output_csv_name_b} 또는 {output_csv_name_w} 은 이미 존재하는 csv 데이터입니다')
        quit()
    f_csv_b = open(output_csv_name_b, 'w', encoding='utf-8',newline='')
    f_csv_w = open(output_csv_name_w, 'w', encoding='utf-8',newline='')

    for count in range(len(file_list)):
        if not (file_list[count].endswith('.REC') or file_list[count].endswith('.rec') or file_list[count].endswith('.psq')):
            print(f'잘못된 확장자 : {file_list[count]}')
            quit()

    b_states_list = []
    w_states_list = []
    b_labels_list = []
    w_labels_list = []
    all_states = np.zeros([15, 15])

    for count in range(len(file_list)):
        data_file_name = file_list[count]
        f = open(read_folder+ '/' + data_file_name, 'r')
        list = []
        skip_n = skip_count
        no_add_csv_count = 0
        while True:
            line = f.readline()
            if not line: break
            skip_n = int(skip_n)
            if skip_n > 0:
                skip_n -= 1
                continue
            list.append(line)

        skip_this = False
        for i in range(len(list)):  # 혹시 15가 넘어가는 파일이 있으면 스킵
            split = list[i].split(",")
            if len(split) != 2:
                print("error : i =", i)
                quit()
            try:
                x = int(split[0])
                y = int(split[1])
            except:
                print("오류 발생 :",data_file_name)
                quit()

            if x > 15 or y > 15:  # 혹시 판의 크기가 15를 넘어가는 경우
                print(f'{data_file_name} 파일의 판 크기가 15를 넘어갑니다')
                no_add_csv_count += 1
                skip_this = True
                break
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
        output = np.insert(b_states_list[i],0,int(b_labels_list[i]))
        f_csv_writer_b.writerow(output)

    for i in range(len(w_states_list)):
        output = np.insert(w_states_list[i],0,int(w_labels_list[i]))
        f_csv_writer_w.writerow(output)

    print("작성 완료 : 스킵한 데이터 ",no_add_csv_count)
    f_csv_b.close()
    f_csv_w.close()


if __name__ == '__main__':
    argstr = input("folder name / skip_count : ")
    args = argstr.split(" ")
    convert(args[0],args[1])
