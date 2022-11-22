# csv 형태로 변경
# https://gomocup.org/results/ 데이터 셋 변경용

import copy
import math
import csv
from datetime import time, datetime
import numpy as np
import os


# arr : numpy
# 0이 아닌 개수
from renju_rule import Renju_Rule


def get_all_stone_count(arr):
    return np.count_nonzero(arr)

def check_right(arr_list,y,x,stone):
    for s in range(x,x+5):
        if s >= 15: return False # 좌표 초과
        if arr_list[y][s] == stone:
            continue
        else:
            return False # 하나라도 안되면 탈락
    return True

def check_down(arr_list,y,x,stone):
    for s in range(y,y+5):
        if s >= 15: return False # 좌표 초과
        if arr_list[s][x] == stone:
            continue
        else:
            return False # 하나라도 안되면 탈락
    return True

def check_down_cross_right(arr_list,y,x,stone):
    cnt = 0
    while True:
        if y+cnt>= 15 or x+cnt >=15:
            return False
        elif arr_list[y+cnt][x+cnt] == stone:
            cnt+=1
            if cnt == 5: return True
        else:
            return False

def check_down_cross_left(arr_list,y,x,stone):
    cnt = 0
    while True:
        if y+cnt< 0 or x-cnt <0 or y+cnt>=15 or x-cnt>=15:
            return False
        elif arr_list[y+cnt][x-cnt] == stone:
            cnt+=1
            if cnt == 5: return True
        else:
            return False



# draw면 0, 흑이 이긴거면 1, 백이 이긴거면 2, 예측 불가면 3
def contains_five(arr_list,size):
    for y in range(size):
        for x in range(size):
            if arr_list[y][x] == 1:
                if check_right(arr_list,y,x,1) or check_down_cross_right(arr_list,y,x,1) \
                        or check_down_cross_left(arr_list,y,x,1) or check_down(arr_list,y,x,1):
                    return 1
            elif arr_list[y][x] == 2:
                if check_right(arr_list,y,x,2) or check_down_cross_right(arr_list,y,x,2)\
                        or check_down_cross_left(arr_list,y,x,2) or check_down(arr_list,y,x,2):
                    return 2
    return 3


def who_is_winner(arr,rule): # 15x15 형태의 numpy array
    if rule == 'renju':
        width = 15
        height = 15
        all_stone_count = get_all_stone_count(arr)
        arr_list = list(arr)
        if all_stone_count < 9: # all_stone_count : 백이랑 흑 합쳐서 전체 돌 수. 이게 만약 9보다 작으면 잘못된 게임
            return 3
        elif all_stone_count == width*height:
            return 0
        else:
            # rule = Renju_Rule(arr_list, width)
            return contains_five(arr_list,width)
    else:
        print("아직 다른 룰은 누가 승자인지 판별 불가능")
        quit()
    return None


def convert_to_label(x, y):
    if x == 0 or y == 0:
        print("시작이 0인 데이터가 존재합니다")
        quit()
    return 15 * (y - 1) + (x - 1)


def convert(folder_name):
    # 파일 헤더
    # 다음 이동 좌표 / black_value / white_value

    board_size=15
    rule = 'renju'
    read_folder = folder_name
    current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    output_csv_name = f'csv_data/output_{current_time}.csv'
    # output_csv_name_w = f'csv_data/white/output_{current_time}_w.csv'

    print("파일을 읽는 중입니다..")
    if not os.path.isdir(folder_name):
        print(f'\"{folder_name}\"은 존재하지 않는 폴더입니다)')

    file_list = []
    for (root, directories, files) in os.walk(read_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    if os.path.isfile(output_csv_name) :
        print(f'{output_csv_name}은 이미 존재하는 csv 데이터입니다')
        quit()
    f_csv = open(output_csv_name, 'w', encoding='utf-8', newline='')

    winner_move_labels = []  # 승자가 이기는 move
    values_black = []  # 0 or 1 # 백은 따로 계산 X (흑의 확률에서 -1)
    turn_stone = [] # 누구 차례인지 (1이면 흑, 2면 백)

    count_not_dataset_file = 0  # 확장자가 dataset 파일이 아닌 경우 (rec, psq 파일이 아닌 경우)
    count_not_1515 = 0  # 15x15가 아니거나 시작 좌표가 0인 경우
    file_list_len = len(file_list)
    print_count = int(file_list_len / 100)  # 1%마다 출력
    next_print_count = print_count

    black_win_count = 0 # (디버그용)
    white_win_count = 0 # (디버그용)
    draw_count = 0 # (디버그용)
    game_count = 0
    unknown_win_count = 0 # (디버그용) 누가 이겼는지 판단이 안되는 경우
    unknown_sample = [] # (디버그용) 승리 알 수 없는 데이터 중 일부
    states_list = []  # state

    print(f"변환 할 전체 파일 수 : {file_list_len}")

    for count in range(len(file_list)):  # 파일 하나당 하나의 반복 (이 for문 한번이 한 게임)
        data_file_name = file_list[count]
        # req, psq 파일이 아닌 경우
        if not (data_file_name.endswith('.REC') or data_file_name.endswith('.rec') or data_file_name.endswith('.psq')):
            count_not_dataset_file += 1
            continue

        f = open(data_file_name, 'r')
        list = []  # 실제 좌표가 포함된 라인
        skip_this = False
        while True:
            line = f.readline()
            if not line: break
            if not ',' in line: # 쉼표가 없는 경우
                continue
            split = line.split(",")
            if not (2 <= len(split) <= 3):  # n,n,n 형태 또는 n,n형태가 아니면 스킵
                continue
            try:
                x = int(split[0])
                y = int(split[1])
                # 혹시 판의 크기가 15를 넘어가는 경우에는 아예 사용하지 않는 파일
                if x > 15 or y > 15 or x == 0 or y == 0:
                    # print(f'{data_file_name} 파일의 판 크기에서 발견된 데이터 : ({x},{y}) (0으로 시작하거나 15를 넘는 데이터)')
                    count_not_1515 += 1
                    skip_this = True
                    break
                else:  # 정상적인 좌표 데이터인 경우
                    list.append(line)
            except:  # split이 안된다면 continue
                continue

        if skip_this: continue
        else : game_count+=1

        all_states = np.zeros([15, 15])  # 하나의 파일이 끝나면 판 초기화
        # 주의!! 하지만, 훈련 데이터들이 비어있는 데이터로 시작하는게 아니라 흑이 먼저 하나 놓은 상태로 시작한다는 점
        for i in range(len(list)):  # 이건 "한번의 게임"에 들어 있는 좌표 데이터 line
            split = list[i].split(",")
            x = int(split[0])
            y = int(split[1])

            if i % 2 == 0:  # 흑
                turn_stone.append(1)
                states_list.append(all_states)
                label = convert_to_label(x, y)
                winner_move_labels.append(label)
                t = copy.deepcopy(all_states)
                t[y - 1][x - 1] = 1
                all_states = t
            else:  # 백
                turn_stone.append(2)
                states_list.append(all_states)
                label = convert_to_label(x, y)
                winner_move_labels.append(label)
                t = copy.deepcopy(all_states)
                t[y - 1][x - 1] = 2
                all_states = t

        winner_number = who_is_winner(arr=all_states,rule=rule)  # draw면 0, 흑이 이긴거면 1, 백이 이긴거면 2, 예측 불가면 3
        if winner_number == 0: # 무승부
            black_value = 50
            draw_count += 1
        elif winner_number == 1: # 흑이 승리
            black_value = 100
            black_win_count += 1
        elif winner_number == 2:  # 백이 승리
            black_value = 0
            white_win_count += 1
        elif winner_number == 3:
            black_value = -1000  # 나중에 가치망에서 학습 할 때 제외
            unknown_win_count += 1
            if unknown_win_count < 3:
                unknown_sample.append(copy.deepcopy(all_states))
        else:
            print(winner_number+"이 0,1,2,3이 아닙니다")
            quit()

        for i in range(len(list)):
            values_black.append(black_value)

        # 데이터가 잘못들어간 경우
        if (len(values_black) != len(states_list)) or (len(values_black) != len(turn_stone)):
            print(f'데이터 오류 : values_black : {len(values_black)} / turn_stone : {len(turn_stone)} / all_states {len(all_states)}')
            quit()
        else: len_state_count = len(values_black)

        # 몇 퍼센트 진행되었는지 출력
        if count >= next_print_count:
            next_print_count += print_count
            percent = round((count / file_list_len) * 100)
            print(f"변환 진행률 : {percent}%")


    # 리스트에 다 넣은 경우
    f_csv_writer = csv.writer(f_csv)
    print("csv 파일 작성 시작..")
    # header_np = np.array(['next_move','black_value','white_value'])
    # f_csv_writer.writerow()
    for i in range(len_state_count):
        output = np.insert(states_list[i], 0, int(turn_stone[i]))
        output = np.insert(output, 0, int(values_black[i]))
        output = np.insert(output, 0, int(winner_move_labels[i]))
        f_csv_writer.writerow(output)
        if i % 5000 == 0:
            print(f"({i}/{len_state_count}) 개 저장..")

    print("\n---------------변환 성공---------------\n")
    print(f"전체 파일 수 : {file_list_len}")
    print(f'스킵한 전체 파일 수 : {count_not_dataset_file + count_not_1515}')
    print(f"데이터셋 확장자가 아닌 파일 수 : {count_not_dataset_file}")
    print(f"다른 형식 데이터셋 파일 :  {count_not_1515}")
    print(f'전체 게임 수 : {game_count}')
    print(f"흑 승리 :  {black_win_count}")
    print(f"백 승리 :  {white_win_count}")
    print(f"무승부 :  {draw_count}")
    print(f"승리 판별 불가 :  {unknown_win_count}")
    print(f'판별 불가 상태 샘플 : {unknown_sample}')
    print(f'파일 경로 : {output_csv_name}')
    f_csv.close()


if __name__ == '__main__':
    argstr = input("folder name : ")
    args = argstr.split(" ")
    convert(args[0])
