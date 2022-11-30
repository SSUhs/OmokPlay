# train set을 사용하는 플레이어
import random

import numpy as np
from time import time
import tensorflow as tf
import keras.backend as K
import copy

_play_on_colab = False
_test_mode = False


# 금수 or 이미 수가 놓아지지 않은 자리 중에서 가장 최선의 인덱스
def get_best_idx(probs, board, is_human_intervene, size=15):
    probs_tmp = copy.deepcopy(probs)  # 아마 ndarray(1,225)??
    # if self.is_test_mode:
    #     self.print_states_probs(probs_tmp)

    # 특수상황에서 사람의 알고리즘 강제 개입 하는 경우
    if is_human_intervene:
        # 먼저, 놓자마자 바로 이길 수 있는 좌표가 있으면 해당 좌표를 선택하면 된다
        can_win_list = get_win_list(board, True)  # 바로 이길 수 있는 위치 확인 (0~224 1차원 좌표)

        if _test_mode:
            print('------------------------------')
            print(f"이길 수 있는 좌표 리스트 : {can_win_list}")
        if len(can_win_list) >= 1:
            return random.choice(can_win_list)  # 어차피 리스트에 있는거 아무거나 놔도 이기므로 하나 랜덤으로 골라서 리턴

        # 이제 상대가 놓으면 바로 이길 수 있는 자리 탐색
        # can_lost_list나 can_win_list는 금수는 이미 처리하고 리턴됨
        can_lose_list = get_win_list(board, False)  # 상대 입장에서 이기는 거 테스트 (type : (0~224 1차원 좌표))
        if _test_mode: print(f'질 수 있는 좌표 리스트 : {can_lose_list}')
        if len(can_lose_list) >= 1:  # 만약 존재한다면, 둘중에 probs가 높은 쪽을 막도록 작동
            arr_tmp = probs_tmp[0][can_lose_list]
            best_choice_idx = np.argmax(arr_tmp)
            best_move = can_lose_list[best_choice_idx]
            return best_move

        arr_list = board.states_loc

        can_attack_list_43 = get_next_43(size, arr_list, board, is_my_turn=True)  # 확실한 공격이 가능한 경우
        can_attack_list_open4 = get_next_open4(size, arr_list, board, is_my_turn=True)  #
        can_attack_list_33 = []
        if _test_mode:
            print(f"공격 가능 43 : {can_attack_list_43}")
            print(f"공격 가능 4open : {can_attack_list_open4}")
        if board.is_you_white():
            can_attack_list_33 = get_next_33(size, arr_list, board, is_my_turn=True)
            if _test_mode: print(f"공격 가능 33 : {can_attack_list_33}")
        can_attack_list = can_attack_list_43 + can_attack_list_33 + can_attack_list_open4
        if len(can_attack_list) >= 1:
            arr_tmp = probs_tmp[0][can_attack_list]
            best_choice_idx = np.argmax(arr_tmp)
            best_move = can_attack_list[best_choice_idx]
            return best_move

        can_defend_list_43 = get_next_43(size, arr_list, board, is_my_turn=False)  # 상대가 나에게 확실한 공격이 가능한 경우
        can_defend_list_4 = get_next_open4(size, arr_list, board, is_my_turn=False)
        if _test_mode:
            print(f"방어 필요 43 : {can_defend_list_43}")
            print(f"방어 필요 4open : {can_defend_list_4}")
        can_defend_list_33 = []
        if board.is_you_black():
            can_defend_list_33 = get_next_33(size, arr_list, board, is_my_turn=False)
            if _test_mode: print(f"방어 필요 33 : {can_defend_list_33}")
        can_defend_list = can_defend_list_43 + can_defend_list_33 + can_defend_list_4
        if len(can_defend_list) >= 1:
            arr_tmp = probs_tmp[0][can_defend_list]
            best_choice_idx = np.argmax(arr_tmp)
            best_move = can_defend_list[best_choice_idx]
            return best_move

    # 특수 알고리즘에 해당 안되면 최대 확률 부분을 찾는다
    while True:
        best_index = np.argmax(probs_tmp[0])
        # 이미 돌이 있는 자리를 선택하거나 금수에 놓은 경우
        if is_banned_pos(board, best_index):
            probs_tmp[0][best_index] = -1  # 금수 자리는 선택 불가능 하게 설정
            continue
        else:
            break
    return best_index


# 현재 상태에서 놓으면 바로 이길 수 있는 위치 찾기
# my_turn : True면 현재 상태에서 내가 놓을 거, False면 상대의 경우로 따지는 것
#           (my_turn을 False로 하면, 상대 입장에서 놓으면 이길 수 있는 리스트를 획득 가능)
def get_win_list(board, my_turn, size=15):
    is_black = board.is_you_black()
    stone = None  # 흑이면 1 백이면 2
    if my_turn:  # 내 차례인데 is_black >> "흑"
        if is_black:
            stone = 1
        else:
            stone = 2
    else:  # 내 차례가 아닌데 예측하는 경우
        if is_black:
            stone = 2
        else:
            stone = 1

    # arr_list : ndarray(15,15) 형태를 list()로
    arr_list = board.states_loc

    win_list = []  # 0~224의 좌표
    for y in range(size):
        for x in range(size):
            if arr_list[y][x] == stone:
                tf, x_new, y_new = check_right_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

                tf, x_new, y_new = check_left_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

                tf, x_new, y_new = check_down_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

                tf, x_new, y_new = check_up_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

                tf, x_new, y_new = check_down_cross_right_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

                tf, x_new, y_new = check_up_cross_left_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

                tf, x_new, y_new = check_down_cross_left_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

                tf, x_new, y_new = check_up_cross_right_5_can_win(arr_list, y, x, stone)
                if tf: win_list.append(convert_xy_to_1nd(x_new, y_new, size))

    return_list = []
    # 안되는 좌표인지 체크
    for i in range(len(win_list)):
        if not is_banned_pos(board, win_list[i], my_turn):  # 내 차례를 보는건지 상대 차례를 보는건지
            return_list.append(win_list[i])
    return return_list


def load_model_trainset_mode(model_type, size, train_num):
    model = None
    model_file = None
    if model_type == 'policy':
        model_file = f'./model_train/tf_policy_{size}_{train_num}.h5'  # 현재 흑백 통합
        model = tf.keras.models.load_model(model_file)
    elif model_type == 'value':
        model_file = f'./model_train/tf_value_{size}_{train_num}.h5'  # 현재 흑백 통합
        model = tf.keras.models.load_model(model_file, compile=False)
    else:
        print("잘못된 타입")
        quit()
    return model

def load_model_train_set_github(model_type, size):
    model = None
    model_file = None
    if model_type == 'policy':
        # model_file = f'./model_train/tf_policy_{size}_{train_num}_{black_white_ai}.h5'
        model_file = f"./OmokPlay/model/colab_policy.h5"
        model = tf.keras.models.load_model(model_file)
    elif model_type == 'value':
        model_file = f"./OmokPlay/model/colab_value.h5"
        model = tf.keras.models.load_model(model_file, compile=False)
    else:
        print("잘못된 타입")
        quit()
    return model


def convert_to_one_dimension(state):
    return np.concatenate(state)


def reshape_to_15_15_1(data):
    return K.reshape(data, [-1, 15, 15, 1])


# 1차원 좌표를 2차원으로
# 시작 좌표 : (0,0)
# 15x15
def convert_xy_to_1nd(x, y, board_size):
    return board_size * y + x


# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (좌우 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_right_5_can_win(arr_list, y, x, stone, reverse_dir=False):
    length = 0  # 이게 4가 되어야 5개중 4개가 놓인 상태이므로 이기는 것
    not_count = 0
    ans_x = None
    ans_y = None
    stone_reverse = None
    if stone == 1:
        stone_reverse = 2
    else:
        stone_reverse = 1
    gap = 6
    add = 1
    if reverse_dir:
        gap = -6
        add = -1

    for s in range(x, x + gap, add):  # 6목도 고려해서 6까지
        if s >= 15 or s < 0:  # 좌표 초과
            break
        if arr_list[y][s] == stone:
            length += 1
            continue
        elif arr_list[y][s] == stone_reverse:  # 상대 돌 만나면 종료
            not_count += 1
            break
        else:  # 놓아야 할 자리
            if not_count == 0:
                ans_x = s
                ans_y = y
            not_count += 1
    if length == 4 and not_count <= 2 and (ans_x is not None):
        return True, ans_x, ans_y
    elif length == 5 and stone == 2 and not_count == 1 and (ans_x is not None):  # 백이면서 6개중에 1개만 비어있고, 하나 놓으면 6목이 되는 경우
        return True, ans_x, ans_y
    else:
        return False, None, None


def check_left_5_can_win(arr_list, y, x, stone):
    return check_right_5_can_win(arr_list, y, x, stone, reverse_dir=True)


# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (아래 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_down_5_can_win(arr_list, y, x, stone, reverse_dir=False):
    length = 0
    not_count = 0
    ans_x = None
    ans_y = None
    end_for = y + 6
    stone_reverse = None
    if stone == 1:
        stone_reverse = 2
    else:
        stone_reverse = 1
    add = 1
    if reverse_dir:
        end_for = y - 6
        add = -1
    for s in range(y, end_for, add):
        if s >= 15 or s < 0:
            break
        if arr_list[s][x] == stone:
            length += 1
            continue
        elif arr_list[s][x] == stone_reverse:  # 상대 돌 만나면 종료
            not_count += 1
            break
        else:
            if not_count == 0:
                ans_x = x
                ans_y = s
            not_count += 1
    if length == 4 and not_count <= 2 and (ans_x is not None):  # 흑 or 백인 상황에서 5개중 1개만 비어있는 경우 하나 놓으면 이김
        return True, ans_x, ans_y
    elif length == 5 and stone == 2 and not_count == 1 and (ans_x is not None):  # 백이면서 6개중에 1개만 비어있고, 하나 놓으면 6목이 되는 경우
        return True, ans_x, ans_y
    else:
        return False, None, None


# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (대각선 오른쪽 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_down_cross_right_5_can_win(arr_list, y, x, stone, reverse_dir=False):
    length = 0
    not_count = 0
    ans_x = None
    ans_y = None
    stone_reverse = None
    if stone == 1:
        stone_reverse = 2
    else:
        stone_reverse = 1
    for i in range(6):
        if reverse_dir: i = - i
        if y + i >= 15 or x + i >= 15 or x + i < 0 or y + i < 0:  # 좌표 초과
            break
        elif arr_list[y + i][x + i] == stone:
            length += 1
        elif arr_list[y + i][x + i] == stone_reverse:
            not_count += 1
            break
        else:  # 돌이 없을 때 : 해당 위치가 놓을 자리
            if not_count == 0:
                ans_x = x + i
                ans_y = y + i
            not_count += 1
        if reverse_dir: i = - i
    if length == 4 and not_count <= 2 and (ans_x is not None):
        return True, ans_x, ans_y
    elif length == 5 and stone == 2 and not_count == 1 and (ans_x is not None):
        return True, ans_x, ans_y
    else:
        return False, None, None


# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (대각선 왼쪽 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_down_cross_left_5_can_win(arr_list, y, x, stone, reverse_dir=False):
    length = 0
    not_count = 0
    ans_x = None
    ans_y = None
    stone_reverse = None
    if stone == 1:
        stone_reverse = 2
    else:
        stone_reverse = 1
    for i in range(6):
        if reverse_dir: i = -i
        if y + i < 0 or y - i < 0 or y + i >= 15 or x - i >= 15:  # 좌표 초과
            break
        elif arr_list[y + i][x - i] == stone:
            length += 1
        elif arr_list[y + i][x - i] == stone_reverse:
            not_count += 1
            break
        else:  # 돌이 없을 때 : 해당 위치가 놓을 자리
            if not_count == 0:
                ans_x = x - i
                ans_y = y + i
            not_count += 1
        if reverse_dir: i = -i
    if length == 4 and not_count <= 2 and (ans_x is not None):
        return True, ans_x, ans_y
    elif length == 5 and stone == 2 and not_count == 1 and (ans_x is not None):
        return True, ans_x, ans_y
    else:
        return False, None, None


def check_up_cross_right_5_can_win(arr_list, y, x, stone):
    return check_down_cross_left_5_can_win(arr_list, y, x, stone, reverse_dir=True)


def check_up_cross_left_5_can_win(arr_list, y, x, stone):
    return check_down_cross_right_5_can_win(arr_list, y, x, stone, reverse_dir=True)


def check_up_5_can_win(arr_list, y, x, stone):
    return check_down_5_can_win(arr_list, y, x, stone, reverse_dir=True)


def get_enemy_stone(stone):
    if stone == 1:
        return 2
    else:
        return 1


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 이미 수가 놓아진 자리 or 금수 자리
# index : 0~224의 1차원화 좌표
# my_turn : 내 차례가 아니라 상대 차례를 예측 하는 경우, 흑백을 뒤집어야함
def is_banned_pos(board, index, my_turn=True):
    is_black = None
    if my_turn:
        is_black = board.is_you_black()
    else:
        is_black = not board.is_you_black()
    if (index in board.states) or (is_black and (index in board.forbidden_moves)):
        return True
    else:
        return False


# dir 1 : 오른쪽
# dir -1 : 왼쪽
# dir 2 : 위쪽
# dir -2 : 아래쪽
# dir 3 : 대각 오른쪽 위
# dir -3 : 대각 왼쪽 아래
# dir 4 : 대각 오른쪽 아래
# dir -4 : 대각 왼쪽 위
# (x,y)에서 dir방향으로 이동 했을 때 어떤 돌인지 (-1 : 막혀있는 공간 / 0 : 빈공간 / 1 : 흑 / 2 : 백)
# diff 증분
# ex) diff = 1이고 dir이 -1이면 오른쪽  diff = -1이고 dir이 1이여도 동일
def gnd(size, arr_list, y, x, diff, dir):
    if diff < 0 and dir > 0:
        diff = -diff
        dir = -dir
    elif diff < 0 and dir < 0:
        print("diff와 dir이 둘다 음수일 수는 없습니다")
        quit()

    if dir == 1:  # 오른쪽
        if x + diff >= size: return -1
        return arr_list[y][x + diff]
    elif dir == -1:  # 왼쪽
        if x - diff < 0: return -1
        return arr_list[y][x - diff]
    elif dir == 2:  # 위쪽
        if y - diff < 0: return -1
        return arr_list[y - diff][x]
    elif dir == -2:  # 아래쪽
        if y + diff >= size: return -1
        return arr_list[y + diff][x]
    elif dir == 3:  # 대각 오른쪽 위
        if y - diff < 0 or x + diff >= size: return -1
        return arr_list[y - diff][x + diff]
    elif dir == -3:  # 대각 왼쪽 아래
        if y + diff >= size or x - diff < 0: return -1
        return arr_list[y + diff][x - diff]
    elif dir == 4:  # 대각 오른쪽 아래
        if y + diff >= size or x + diff >= size: return -1
        return arr_list[y + diff][x + diff]
    elif dir == -4:  # 대각 왼쪽 위
        if y - diff < 0 or x - diff < 0: return -1
        return arr_list[y - diff][x - diff]
    else:
        print("존재하지 않는 방향")
        quit()


def get_dir_move_1nd(size, arr_list, y, x, diff, dir):
    new_x = x
    new_y = y
    if dir == 1:  # 오른쪽
        new_x += diff
    elif dir == -1:  # 왼쪽
        new_x -= diff
    elif dir == 2:  # 위쪽
        new_y -= diff
    elif dir == -2:  # 아래쪽
        new_y += diff
    elif dir == 3:  # 대각 오른쪽 위
        new_x += diff
        new_y -= diff
    elif dir == -3:  # 대각 왼쪽 아래
        new_x -= diff
        new_y += diff
    elif dir == 4:  # 대각 오른쪽 아래
        new_x += diff
        new_y += diff
    elif dir == -4:  # 대각 왼쪽 위
        new_x -= diff
        new_y -= diff
    else:
        print("존재하지 않는 방향")
        quit()
    return convert_xy_to_1nd(new_x, new_y, size)


# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 열린 3이 되는 좌표 "리스트" 리턴
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def get_next_open3(size, arr_list, board, is_my_turn):
    list_move = []
    empty = 0
    stone = None
    if is_my_turn:
        if board.is_you_black():
            stone = 1
            enemy_stone = 2
        else:
            stone = 2
            enemy_stone = 1
    else:
        if board.is_you_black():
            stone = 2
            enemy_stone = 1
        else:
            stone = 1
            enemy_stone = 2
    # enemy_stone = get_enemy_stone(stone) # 흰색이면 백
    for y in range(size):
        for x in range(size):
            if arr_list[y][x] == stone:
                for dir in range(-4, 5, 1):  # -4번부터 5번까지 (단 0은 continue)
                    if dir == 0:
                        continue
                    if gnd(size, arr_list, y, x, 1, dir) == stone \
                            and gnd(size, arr_list, y, x, 1, -dir) == empty \
                            and gnd(size, arr_list, y, x, 2, dir) == empty \
                            and gnd(size, arr_list, y, x, 3, dir) == empty:
                        list_move.append(get_dir_move_1nd(size, arr_list, y, x, 2, dir))

                    if gnd(size, arr_list, y, x, 1, dir) == empty \
                            and gnd(size, arr_list, y, x, 1, -dir) == empty \
                            and gnd(size, arr_list, y, x, 2, dir) == stone \
                            and gnd(size, arr_list, y, x, 3, dir) == empty:
                        list_move.append(get_dir_move_1nd(size, arr_list, y, x, 1, dir))
    return list_move


# 설명은 get_next_open3 확인
def get_next_open4(size, arr_list, board, is_my_turn):
    list_move = []
    empty = 0
    stone = None
    if is_my_turn:
        if board.is_you_black():
            stone = 1
        else:
            stone = 2
    else:
        if board.is_you_black():
            stone = 2
        else:
            stone = 1

    # enemy_stone = get_enemy_stone(stone) # 흰색이면 백
    for y in range(size):
        for x in range(size):
            if arr_list[y][x] == stone:
                for dir in range(-4, 5, 1):  # -4번부터 5번까지 (단 0은 continue)
                    if dir == 0:
                        continue
                    s_r1 = gnd(size, arr_list, y, x, 1, -dir)
                    s_1 = gnd(size, arr_list, y, x, 1, dir)
                    s_2 = gnd(size, arr_list, y, x, 2, dir)
                    s_3 = gnd(size, arr_list, y, x, 3, dir)
                    s_4 = gnd(size, arr_list, y, x, 4, dir)
                    if s_r1 == empty:
                        if s_1 == empty and s_2 == stone and s_3 == stone and s_4 == empty:
                            list_move.append(get_dir_move_1nd(size, arr_list, y, x, 1, dir))
                        if s_1 == stone and s_2 == stone and s_3 == empty:
                            list_move.append(get_dir_move_1nd(size, arr_list, y, x, 3, dir))
                        if s_1 == stone and s_2 == stone and s_3 == empty and s_4 == empty:
                            list_move.append(get_dir_move_1nd(size, arr_list, y, x, 2, dir))

    return_list = []
    for i in range(len(list_move)):
        if not is_banned_pos(board, list_move[i], my_turn=is_my_turn):  # 내 차례를 보는건지 상대 차례를 보는건지
            return_list.append(list_move[i])
    return return_list


# 한쪽만 닫힌 경우
def get_next_closed4(size, arr_list, board, is_my_turn):
    list_move = []
    empty = 0
    stone = None
    if is_my_turn:
        if board.is_you_black():
            stone = 1
            enemy_stone = 2
        else:
            stone = 2
            enemy_stone = 1
    else:
        if board.is_you_black():
            stone = 2
            enemy_stone = 1
        else:
            stone = 1
            enemy_stone = 2
    closed = -1  # 공간 넘어가는 부분
    for y in range(size):
        for x in range(size):
            if arr_list[y][x] == enemy_stone or arr_list[y][x] == closed:
                for dir in range(-4, 5, 1):  # -4번부터 5번까지 (단 0은 continue)
                    if dir == 0:
                        continue
                    s_1 = gnd(size, arr_list, y, x, 1, dir)
                    s_2 = gnd(size, arr_list, y, x, 2, dir)
                    s_3 = gnd(size, arr_list, y, x, 3, dir)
                    s_4 = gnd(size, arr_list, y, x, 4, dir)
                    s_5 = gnd(size, arr_list, y, x, 4, dir)
                    if s_1 == stone and s_2 == empty and s_3 == stone and s_4 == stone and s_5 == empty:
                        list_move.append(get_dir_move_1nd(size, arr_list, y, x, 2, dir))
                    if s_1 == empty and s_2 == stone and s_3 == stone and s_4 == stone and s_5 == empty:
                        list_move.append(get_dir_move_1nd(size, arr_list, y, x, 1, dir))
                    if s_1 == stone and s_2 == stone and s_3 == empty and s_4 == stone and s_5 == empty:
                        list_move.append(get_dir_move_1nd(size, arr_list, y, x, 3, dir))
                    if s_1 == stone and s_2 == stone and s_3 == stone and s_4 == empty:
                        list_move.append(get_dir_move_1nd(size, arr_list, y, x, 4, dir))

    return_list = []
    for i in range(len(list_move)):
        if not is_banned_pos(board, list_move[i], is_my_turn):  # 내 차례를 보는건지 상대 차례를 보는건지
            return_list.append(list_move[i])
    return return_list


# move list (0~224)
def get_next_4(size, arr_list, board, is_my_turn):
    list_1 = get_next_open4(size, arr_list, board, is_my_turn)
    list_2 = get_next_closed4(size, arr_list, board, is_my_turn)
    list_3 = list_1 + list_2
    return list_3


# 다음에 놓으면 4,3 이 되는 수
# 어떤 자리에 놓았을 때 (열린4+닫힌4)랑 열린3이 동시에 주어지는 경우
def get_next_43(size, arr_list, board, is_my_turn):
    stone = None
    if is_my_turn:
        if board.is_you_black():
            stone = 1
            enemy_stone = 2
        else:
            stone = 2
            enemy_stone = 1
    else:
        if board.is_you_black():
            stone = 2
            enemy_stone = 1
        else:
            stone = 1
            enemy_stone = 2
    list_4 = get_next_4(size, arr_list, board, is_my_turn)
    list_3_open = get_next_open3(size, arr_list, board, is_my_turn)
    list_ans = list(set(list_4).intersection(list_3_open))
    return_list = []
    for i in range(len(list_ans)):
        if not is_banned_pos(board, list_ans[i], is_my_turn):  # 내 차례를 보는건지 상대 차례를 보는건지
            return_list.append(list_ans[i])
    return list_ans


def get_next_33(size, arr_list, board, is_my_turn):
    stone = None
    if is_my_turn:
        if board.is_you_black():
            stone = 1
            enemy_stone = 2
        else:
            stone = 2
            enemy_stone = 1
    else:
        if board.is_you_black():
            stone = 2
            enemy_stone = 1
        else:
            stone = 1
            enemy_stone = 2
    list_3_open = get_next_open3(size, arr_list, board, is_my_turn)
    set_3 = set(list_3_open)
    list_33 = []
    return_list = []
    if len(set_3) != len(list_3_open):
        tmp_list = list(set_3)
        list_33 = [i for i in list_3_open if i not in tmp_list]

    for i in range(len(list_33)):
        if not is_banned_pos(board, list_33[i], is_my_turn):  # 내 차례를 보는건지 상대 차례를 보는건지
            return_list.append(list_33[i])
    return return_list


def convert_to_2nd_loc(size, index):  # 2차원 좌표로 변경
    y = index // size
    x = index - y * size
    return x, y

    # class player_AI():
    # def __init__(self, size, is_test_mode, black_white_human, train_num, is_sequential_model=True, use_mcts_search=True,is_self_play=False,is_human_intervene=False):
    #     self.size = size
    #     self.is_self_play = is_self_play
    #     self.is_test_mode = is_test_mode
    #     self.black_white_human = black_white_human # 참고로 사람이 흑을 하면 AI는 백을 로딩해야됨
    #     self.black_white_ai = None
    #     if black_white_human == 'black':
    #         self.black_white_ai = 'white'
    #     else:
    #         self.black_white_ai = 'black'
    #
    #     self.model = self.load_model(model_type='policy',black_white_ai=self.black_white_ai, train_num=train_num)
    #     self.is_sequential_model = is_sequential_model
    #     self.value_net_model = self.load_model(model_type='value', black_white_ai=self.black_white_ai,
    #                                            train_num=train_num)
    #     self.is_human_intervene = is_human_intervene  # 사람 알고리즘 개입 (ex : 닫힌 4 무조건 막기 )
    #     self.use_mcts_search = use_mcts_search  # MCTS 검색을 쓸 것인지 아니면 단순히 가장 probs가 높은 걸로 리턴할 것인지
    #     if self.use_mcts_search:
    #         self.mcts = MCTSTrainSet(policy_net=self.model, c_puct=5, n_playout=400, is_test_mode=is_test_mode, value_net=self.value_net_model)
    #         # value_net_tmp = ValueNetTmpNumpy(board_size=size,net_params_file=f'tf_value_{size}_{train_num}_{self.black_white_ai}.pickle')# numpy로 임시로 구현한 가치망
    #
    #
    #
    # def load_model(self, model_type,black_white_ai, train_num):
    #     model = None
    #     model_file = None
    #     if model_type == 'policy':
    #         model_file = f'./model_train/tf_policy_{self.size}_{train_num}_{black_white_ai}.h5'
    #         model = tf.keras.models.load_model(model_file)
    #     elif model_type == 'value':
    #         model_file = f'./model_train/tf_value_{self.size}_{train_num}_{black_white_ai}.h5'
    #         model = tf.keras.models.load_model(model_file,compile=False)
    #     else:
    #         print("잘못된 타입")
    #         quit()
    #
    #     return model

    #
    #
    # # (디버그 용도) 확률 리턴
    # # ndarray_probs_1nd : ndarray(1,225) # 15x15 기준
    # def print_states_probs(self,ndarray_probs_1nd_):
    #     if self.use_mcts_search:
    #         print("mcts는 확률 표 미구현")
    #         return
    #     print("정책망 확률 표 (0,0 부터 시작)")
    #     length = self.size*self.size
    #     ndarray_probs_1nd = copy.deepcopy(ndarray_probs_1nd_)
    #     list_print = []
    #     for i in range(length):
    #         best_idx = np.argmax(ndarray_probs_1nd[0])
    #         best_prob_float = ndarray_probs_1nd[0][best_idx]
    #         ndarray_probs_1nd[0][best_idx] = -1  # 그 다음 최대를 찾기 위해 -1로 수정
    #         if best_prob_float <= 0.001:
    #             continue
    #         x,y = selfconvert_to_2nd_loc(best_idx)
    #         list_print.append(f'({x},{y}) : {format(best_prob_float,".4f")}%')
    #     for i in range(len(list_print)):
    #         print(list_print[i])

    def set_player_ind(self, p):
        self.player = p


class TreeNode(object):
    """ MCTS 트리의 노드.
    Q : its own value
    P : prior probability
    u : visit-count-adjusted prior score
    """

    def __init__(self, parent, prior_p,depth=0):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        self.depth = depth
        # print(f"깊이 : {depth}")

    def expand(self, action_priors, forbidden_moves, is_you_black):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability according to the policy function.
        """
        # action : int 타입
        # action_priors는 zip
        # leag_arr은 크기가 점점 줄어드는 ndarray
        # lega_arr = act_probs[0][legal_positions] # 얘는 수를 놓을 때마다 사이즈가 줄어 듦  # 왜 0번이냐면 애초에 act_probs가 [1][225] 이런형태라 그럼
        # act_probs = zip(legal_positions, lega_arr)

        # 예를들어 {1,2,6,8,13} {0.4231,0.832,~~~} 이런식이라면,
        # 현재 노드에서 1번으로 확장하면 해당 노드의 가중치는 0.4231이 되는 것
        for_count = 0
        for action, prob in action_priors:
            for_count +=1
            if is_you_black and action in forbidden_moves:
                continue
            if action not in self._children:  #  code20221130141219
                if prob < 0.001: # 확률이 너무 낮은 부분은 확장하지 않음
                    continue
                self._children[action] = TreeNode(self, prob,self.depth+1)
        print(f'for_count : {for_count}')

    def select(self, c_puct):
        # 자식 노드 중에서 가장 적절한 노드를 선택 한다 (action값)
        """Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's perspective.
        리프 노드까지 다 진행 후에 무승부가 나는 경우 leaf_value가 0이 되고, 패배하면 -1이 되고, 이기면 1이 된다
        https://glbvis.blogspot.com/2021/05/ai_20.html
        여기 중간 그림 보면 Terminal State에서 1/-1 나와 있다
        (update_recursive()를 수행할 때 leaf_value에다가 양음 바꿔서 처리)
        """
        # 방문 횟수 체크 (평균 계산을 위해서 방문 노드 수 체크)
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        # leaf_value 타입 : ndarray[1,1]
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 자식 노드부터 부모 노드까지 가치값 업데이트
    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors."""
        # If it is not root, this node's parent should be updated first.
        # if 뒤에 객체가 오는 경우 : __bool__이 오버라이딩 되어 있지 않다면, None이면 false리턴
        # 따라서 아래의 조건문을 만족 시키는 경우, 부모 노드가 존재하는 것이므로 부모 노드부터 업데이트 수행
        # 아래 조건문이 false라면 부모 노드가 없는 노드이므로 root 노드
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        # leaf node : 자식 노드가 없는 노드
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTSTrainSet(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_net, c_puct=5, n_playout=10000, is_test_mode=False, value_net=None):
        """
        policy_value_fn: a function that takes in a board_img state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self.policy_net = policy_net
        self.value_net = value_net
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.is_test_mode = is_test_mode

    # state : 현재 상태에서 deepcopy 된 state
    # 이 함수는 사용자와의 대결에도 사용 된다
    # 각 상태에서 끝까지 플레이 해본다
    # 이 함수가 n_playout 만큼 돌아가는 것 (디폴트 : 400번)
    def _playout(self, state, black_white_ai):
        node = self._root
        while (1):
            # 리프 노드가 나올 때까지 계속 진행
            # 확장은 여기서 안하고 아래 쪽에 node_expand 에서 진행한다
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)  # 리프노드가 나올 때 까지 move

        set_all = set(range(state.width * state.height))
        set_state_keys = set(state.states.keys())
        legal_positions = list(set_all - set_state_keys)
        np_states = state.get_states_by_numpy()
        inputs = reshape_to_15_15_1(np_states)  # 현재 상태. 이 상태를 기반으로 예측
        act_probs = self.policy_net.predict(inputs)
        leaf_value = self.value_net.predict(inputs)[0][0]
        if black_white_ai == 'white':
            leaf_value = -leaf_value
        legal_arr = act_probs[0][legal_positions]  # 얘는 수를 놓을 때마다 사이즈가 줄어 듦  # 왜 0번이냐면 애초에 act_probs가 [1][225] 이런형태라 그럼
        action_probs = zip(legal_positions, legal_arr)
        # end (bool 타입) : 게임이 단순히 끝났는지 안끝났는지 (승,패 또는 화면 꽉찬 경우에도 end = True)
        end, winner = state.game_end()
        if not end:  #
            node.expand(action_probs, state.forbidden_moves, state.is_you_black())
        else:
            # for end state，return the "true" leaf_value
            # winner은 무승부의 경우 -1이고, 우승자가 존재하면 우승자 int (0,1이였나 1,2였나)
            if winner == -1:  # tie (무승부)
                leaf_value = 0.0  # 무승부의 경우 leaf_value를 0으로 조정
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0)  # 우승자가 자신이라면, leaf_value는 1로, 패배자라면 -1로
        node.update_recursive(-leaf_value)

    # 여기서 state는 game.py의 board 객체
    def get_move_probs(self, state, black_white_ai, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy, black_white_ai)
            print(f"playout : {n}번 수행")

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        # print([(state.move_to_location(m),v) for m,v in act_visits])

        # acts = 위치번호 / visits = 방문횟수
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        if self.is_test_mode:
            print(f'acts : {acts} / size {len(acts)}')
            print(f'visits : {visits} / size {len(visits)}')

        return acts, act_probs

    # 플레이어 대결의 경우 컴퓨터는 update_with_move를 호출 할 때 last_move 파라미터를 -1로 전달 (아직 무슨 의미인지는 파악 X)
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]  # 돌을 둔 위치가 root노드가 됨
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer_TrainSet(object):
    def __init__(self, policy_net, value_net,
                 c_puct=5, n_playout=2000, is_selfplay=0, is_test_mode=False,
                 is_human_intervene=True, black_white_ai=None,use_mcts=True):
        # 여기서 policy_value_function을 가져오기 때문에 어떤 라이브러리를 선택하냐에 따라 MCTS속도가 달라짐
        self.mcts = MCTSTrainSet(policy_net, c_puct, n_playout, is_test_mode=is_test_mode, value_net=value_net)
        self.policy_net = policy_net
        self.value_net = value_net
        self.is_human_intervene = is_human_intervene
        self._is_selfplay = is_selfplay
        self.is_test_mode = is_test_mode
        self.black_white_ai = black_white_ai
        self.use_mcts = use_mcts

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    # n*n 형태를 일차원으로
    def get_action(self, board, black_white_ai):
        # state : numpy
        state = board.get_states_by_numpy()
        if self.use_mcts:
            move, value = self.get_move_mcts(board, black_white_ai)  # mcts를 사용해서 추가 예측
        else:
            inputs = reshape_to_15_15_1(state)  # 현재 상태. 이 상태를 기반으로 예측
            move, value = self.get_move_not_mcts(board, inputs)  # mcts 없이 단순히 확률이 가장 높은 경우를 선택
        print(f'타입 테스트 : {type(board)} / {board.width} / {type(move)}')
        x, y = convert_to_2nd_loc(board.width, move)
        if _test_mode:
            print(f"선택 좌표 (0,0부터) : {move} = ({x},{y})")
            print(f'가치망 value : {value}')
        return move

    def get_move_not_mcts(self, board, input):
        probs = self.policy_net.predict(input)
        value = self.value_net.predict(input)
        return get_best_idx(probs, board, is_human_intervene=self.is_human_intervene), value

    # MCTS 기반
    def get_move_mcts(self, board, black_white_ai):
        # np.zeros : 0으로만 채워진 배열 생성하는 함수
        size = board.width
        if board.width * board.height - len(board.states) > 0:  # 보드판이 꽉 안찬 경우
            move = None
            inputs = reshape_to_15_15_1(board.get_states_by_numpy())  # 현재 상태. 이 상태를 기반으로 예측
            value_current = self.value_net.predict(inputs)[0]
            if black_white_ai == 'white':  # 백이면 가치망 뒤집는다
                value_current = -value_current
            if self.is_human_intervene:
                # 먼저, 놓자마자 바로 이길 수 있는 좌표가 있으면 해당 좌표를 선택하면 된다

                probs_tmp = self.policy_net.predict(inputs)

                can_win_list = get_win_list(board, True)  # 바로 이길 수 있는 위치 확인 (0~224 1차원 좌표)
                if _test_mode:
                    print('------------------------------')
                    print(f"이길 수 있는 좌표 : {can_win_list}")
                if len(can_win_list) >= 1:
                    return random.choice(can_win_list), 1.0  # 어차피 리스트에 있는거 아무거나 놔도 이기므로 하나 랜덤으로 골라서 리턴

                # 이제 상대가 놓으면 바로 이길 수 있는 자리 탐색
                # can_lost_list나 can_win_list는 금수는 이미 처리하고 리턴됨
                can_lose_list = get_win_list(board, False)  # 상대 입장에서 이기는 거 테스트 (type : (0~224 1차원 좌표))
                if _test_mode: print(f'질 수 있는 좌표  : {can_lose_list}')
                if len(can_lose_list) >= 1:
                    arr_tmp = probs_tmp[0][can_lose_list] # 만약 질 수 있는 위치가 두개 이상이라면, 신경망을 통해 나온 결과중 높은 곳으로 지정
                    best_choice_idx = np.argmax(arr_tmp)
                    best_move = can_lose_list[best_choice_idx]
                    return best_move, value_current

                arr_list = board.states_loc

                can_attack_list_43 = get_next_43(size, arr_list, board, is_my_turn=True)  # 확실한 공격이 가능한 경우
                can_attack_list_open4 = get_next_open4(size, arr_list, board, is_my_turn=True)  #
                if _test_mode:
                    print(f"공격 가능 43 : {can_attack_list_43}")
                    print(f"공격 가능 4open : {can_attack_list_open4}")
                can_attack_list_33 = []
                if board.is_you_white():
                    can_attack_list_33 = get_next_33(size, arr_list, board, is_my_turn=True)
                    if _test_mode: print(f"공격 가능 33 : {can_attack_list_33}")

                can_attack_list = can_attack_list_43 + can_attack_list_33 + can_attack_list_open4
                if len(can_attack_list) >= 1:
                    arr_tmp = probs_tmp[0][can_attack_list]
                    best_choice_idx = np.argmax(arr_tmp)
                    best_move = can_attack_list[best_choice_idx]
                    return best_move, value_current

                can_defend_list_43 = get_next_43(size, arr_list, board, is_my_turn=False)  # 상대가 나에게 확실한 공격이 가능한 경우
                can_defend_list_4 = get_next_open4(size, arr_list, board, is_my_turn=False)
                if _test_mode:
                    print(f"방어 필요 43 : {can_defend_list_43}")
                    print(f"방어 필요 4open : {can_defend_list_4}")
                can_defend_list_33 = []
                if board.is_you_black():
                    can_defend_list_33 = get_next_33(size, arr_list, board, is_my_turn=False)
                    if _test_mode: print(f"방어 필요 33 : {can_defend_list_33}")
                can_defend_list = can_defend_list_43 + can_defend_list_33 + can_defend_list_4
                if len(can_defend_list) >= 1:
                    arr_tmp = probs_tmp[0][can_defend_list]
                    best_choice_idx = np.argmax(arr_tmp)
                    best_move = can_defend_list[best_choice_idx]
                    return best_move, value_current

            # acts와 probs에 의해 착수 위치가 정해진다.
            time_get_probs = time()  # probs를 얻는데까지 걸리는 시간
            acts, probs = self.mcts.get_move_probs(board, black_white_ai)
            # probs = self.model.predict(input)  # 위치별 확률
            if self.is_test_mode:
                time_gap = time() - time_get_probs
                print(f'get_probs 하는데 소요된 시간 : {time_gap}')
            if self._is_selfplay:
                # (자가 학습을 할 때는) Dirichlet 노이즈를 추가하여 탐색
                print("강화 학습은 구현중")
                # move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                # time_update_with_move = time()
                # self.mcts.update_with_move(move)
                # if self.is_test_mode:
                #     print(f'update_with_move 하는데 소요된 시간 : {time() - time_update_with_move}')
            else:  # 플레이어와 대결하는 경우
                move = np.random.choice(acts, p=probs)
                print("mcts ai가 고른 자리 : ", move)
                # 점검
                self.mcts.update_with_move(-1)
            return move, value_current
        else:
            print("WARNING: the board_img is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
