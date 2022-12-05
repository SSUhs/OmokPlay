# 금수 판정 / 승리 판정 등등
import copy
import random
import numpy as np
from rule.renju_rule.renju_rule import Renju_Rule
from constant import error_const


_play_on_colab = False
_test_mode = True

def get_stone_color(black_white_ai):
    if black_white_ai == 'black':
        return 1
    elif black_white_ai == 'white':
        return 2
    else:
        print(f'black_white_ai : {black_white_ai}  : 잘못된 입력')
        quit()



# stone : 흑돌 1 / 백돌 2  (렌주룰)
def is_banned_pos_new(board, index, stone):
    if (index in board.states) or (stone == 1 and (index in board.forbidden_moves)):
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
# "하나 더 놓으면" 열린 3이 되는 좌표 "리스트" 리턴
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def get_next_open3(size, arr_list, board, stone):
    list_move = []
    empty = 0
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
                        list_move.append([get_dir_move_1nd(size, arr_list, y, x, 2, dir),dir])

                    if gnd(size, arr_list, y, x, 1, dir) == empty \
                            and gnd(size, arr_list, y, x, 1, -dir) == empty \
                            and gnd(size, arr_list, y, x, 2, dir) == stone \
                            and gnd(size, arr_list, y, x, 3, dir) == empty:
                        list_move.append([get_dir_move_1nd(size, arr_list, y, x, 1, dir),dir])

                    if gnd(size, arr_list,y,x,1,-dir) == empty \
                            and gnd(size,arr_list,y,x,1,dir) == stone \
                            and gnd(size,arr_list,y,x,2,dir) == empty \
                            and gnd(size, arr_list, y, x, 3, dir) == empty \
                            and gnd(size, arr_list, y, x, 4, dir) == empty:
                        list_move.append([get_dir_move_1nd(size, arr_list, y, x, 3, dir),dir])

                    if gnd(size, arr_list,y,x,1,-dir) == empty \
                            and gnd(size,arr_list,y,x,1,dir) == empty \
                            and gnd(size,arr_list,y,x,2,dir) == stone \
                            and gnd(size, arr_list, y, x, 3, dir) == empty \
                            and gnd(size, arr_list, y, x, 4, dir) == empty:
                        list_move.append([get_dir_move_1nd(size, arr_list, y, x, 3, dir),dir])

                    if gnd(size, arr_list,y,x,1,-dir) == empty \
                            and gnd(size,arr_list,y,x,1,dir) == empty \
                            and gnd(size,arr_list,y,x,2,dir) == empty \
                            and gnd(size, arr_list, y, x, 3, dir) == stone \
                            and gnd(size, arr_list, y, x, 4, dir) == empty:
                        list_move.append([get_dir_move_1nd(size, arr_list, y, x, 1, dir),dir]) # 이건 둘다 추가 가능
                        list_move.append([get_dir_move_1nd(size, arr_list, y, x, 2, dir),dir]) # 이건 둘다 추가 가능

    return_list_tmp = []

    for i in range(len(list_move)):
        escape = False
        # 이미 존재하는 경우 : 위와 아래처럼 완전 반대에서 탐색해서 동일한 좌표가 생긴거라면 추가 X
        for j in range(len(return_list_tmp)):
            if return_list_tmp[j][0] == list_move[i][0] and return_list_tmp[j][1] == -list_move[i][1]:
                escape = True
                break
        if not escape:
            return_list_tmp.append(list_move[i])

    returns = []
    for i in range(len(return_list_tmp)):
        returns.append(return_list_tmp[i][0])

    return returns



# 특정 자리에 놨을 때 일자 리스트
# dir 1 : 좌우
# dir 2 : 상하
# dir 3 : 대각 우상향 (/)
# dir 4 : 대각 우하향 (\)
def get_colors_dir_next(size,arr_list,stone,x,y,dir):
    if arr_list[y][x] != 0:
        print("놓을 자리는 빈자리여야합니다")
        quit()
    elif dir <= 0:
        print("양수 방향만 놓을 수 있습니다")
        quit()
    arr_list[y][x] = stone
    list_stone = []
    list_index = []
    if dir == 1:
        for i in range(0,size):
            list_stone.append(arr_list[y][i])
            list_index.append(convert_xy_to_1nd(i,y,size)) # xy순서 주의
    elif dir == 2:
        for i in range(0,size):
            list_stone.append(arr_list[i][x])
            list_index.append(convert_xy_to_1nd(x,i,size))
    elif dir == 3: # 대각 우상향 (/) (x증가 y감소)
        i = 0
        list_stone1 = []
        list_index1 = []
        list_stone2 = []
        list_index2 = [] # 더 작은 좌표들
        while True:
            list_stone1.append(arr_list[y-i][x+i])
            list_index1.append(convert_xy_to_1nd(x+i,y-i,size))
            i+=1
            if y-i < 0 or x+i >= size:
                break
        i = 1
        while True:
            if y+i >= size or x-i < 0:
                break
            list_stone2.append(arr_list[y+i][x-i])
            list_index2.append(convert_xy_to_1nd(x-i,y+i,size))
            i+=1
        list_stone2.reverse()
        list_index2.reverse()
        list_stone = list_stone2 + list_stone1
        list_index = list_index2 + list_index1
    elif dir == 4: # 대각 우하향 (\)
        i = 0
        list_stone1 = []
        list_index1 = []
        list_stone2 = []
        list_index2 = [] # 더 작은 좌표들
        while True:
            list_stone1.append(arr_list[y+i][x+i])
            list_index1.append(convert_xy_to_1nd(x+i,y+i,size))
            i+=1
            if y+i >= size or x+i >= size:
                break

        i = 1
        while True:
            if y-i < 0 or x-i < 0:
                break
            list_stone2.append(arr_list[y-i][x-i])
            list_index2.append(convert_xy_to_1nd(x-i,y-i,size))
            i+=1
        list_stone2.reverse()
        list_index2.reverse()
        list_stone = list_stone2 + list_stone1
        list_index = list_index2 + list_index1

    arr_list[y][x] = 0 # 다시 지우기
    return list_stone,list_index


# 인자 (x,y)를 놨을 때 열린4가 발생하는 개수
def get_count_open4_when_do(size,arr_list,stone,x,y):
    empty = 0
    # enemy_stone = get_enemy_stone(stone)
    count = 0
    xy_move = convert_xy_to_1nd(x,y,size)
    for dir in range(1,5):
        list_stone,list_index = get_colors_dir_next(size,arr_list,stone,x,y,dir)
        if len(list_stone) < 6: # 6개 미만이면 "열린 4"는 불가능
            return 0
        ans1 = [empty,stone,stone,stone,stone,empty]
        for i in range(len(list_stone)-5):
            list_test = list_stone[i:i+6]
            if list_test == ans1:
                if i <= list_index.index(xy_move) < i+6:
                    count += 1
    return count

# 인자 (x,y)에 놨을 때 4개짜리가 생기는지
def is_4_when_do(size,arr_list,stone,x,y):
    if get_count_closed4_when_do(size,arr_list,stone,x,y) >= 1 or get_count_open4_when_do(size,arr_list,stone,x,y) >= 1:
        return True
    else:
        return False



# 파라미터로 받은 x,y 좌표를 놨을 때 닫힌4가 되는 경우
def get_count_closed4_when_do(size,arr_list,stone,x,y):
    empty = 0
    count = 0
    xy_move = convert_xy_to_1nd(x,y,size)
    for dir in range(1,5):
        list_stone,list_index = get_colors_dir_next(size,arr_list,stone,x,y,dir)
        if len(list_stone) < 6: # 6개 미만이면 "열린 4"는 불가능
            return 0
        ans1 = [stone,stone,stone,stone,empty]
        ans2 = [empty,stone,stone,stone,stone]
        for i in range(len(list_stone)-5):
            list_test = list_stone[i:i+6]
            if list_test == ans1 or list_test == ans2:
                if i <= list_index.index(xy_move) < i+6:
                    count += 1
    return count


# 한쪽만 닫힌 경우
def get_next_closed4(size, arr_list, board, stone):
    list_move = []
    for y in range(size):
        for x in range(size):
            if arr_list[y][x] == 0: # 빈자리인 경우만 탐색
                count = get_count_closed4_when_do(size,arr_list,stone,x,y,)
                if count>=1:
                    list_move.append(convert_xy_to_1nd(x,y,size))
    return_list = []
    for i in range(len(list_move)):
        if not is_banned_pos_new(board, list_move[i], stone):
            return_list.append(list_move[i])
    return return_list

def get_next_open4(size, arr_list, board, stone):
    list_move = []
    for y in range(size):
        for x in range(size):
            if arr_list[y][x] == 0: # 빈자리인 경우만 탐색
                count = get_count_open4_when_do(size,arr_list,stone,x,y)
                if count>=1:
                    list_move.append(convert_xy_to_1nd(x,y,size))
    return_list = []
    for i in range(len(list_move)):
        if not is_banned_pos_new(board, list_move[i], stone):
            return_list.append(list_move[i])
    return return_list


# 새로운 방식으로 변경
# def get_next_open4(size, arr_list, board, stone):
#     list_move = []
#     empty = 0
#
#     # enemy_stone = get_enemy_stone(stone) # 흰색이면 백
#     for y in range(size):
#         for x in range(size):
#             if arr_list[y][x] == stone:
#                 for dir in range(-4, 5, 1):  # -4번부터 5번까지 (단 0은 continue)
#                     if dir == 0:
#                         continue
#                     s_r1 = gnd(size, arr_list, y, x, 1, -dir)
#                     s_1 = gnd(size, arr_list, y, x, 1, dir)
#                     s_2 = gnd(size, arr_list, y, x, 2, dir)
#                     s_3 = gnd(size, arr_list, y, x, 3, dir)
#                     s_4 = gnd(size, arr_list, y, x, 4, dir)
#                     if s_r1 == empty:
#                         if s_1 == empty and s_2 == stone and s_3 == stone and s_4 == empty:
#                             list_move.append(get_dir_move_1nd(size, arr_list, y, x, 1, dir))
#                         if s_1 == stone and s_2 == stone and s_3 == empty:
#                             list_move.append(get_dir_move_1nd(size, arr_list, y, x, 3, dir))
#                         if s_1 == stone and s_2 == stone and s_3 == empty and s_4 == empty:
#                             list_move.append(get_dir_move_1nd(size, arr_list, y, x, 2, dir))
#
#     return_list = []
#     for i in range(len(list_move)):
#         if not is_banned_pos_new(board, list_move[i], stone):  # 내 차례를 보는건지 상대 차례를 보는건지
#             return_list.append(list_move[i])
#     return return_list

# move list (0~224)
def get_next_4(size, arr_list, board, stone):
    list_1 = get_next_open4(size, arr_list, board, stone)
    list_2 = get_next_closed4(size, arr_list, board, stone)
    list_3 = list_1 + list_2
    return list_3


# 다음에 놓으면 4,3 이 되는 수
# 어떤 자리에 놓았을 때 (열린4+닫힌4)랑 열린3이 동시에 주어지는 경우
# def get_next_43(size, arr_list, board, stone):
#
#     list_4 = get_next_4(size, arr_list, board, stone)
#     list_3_open = get_next_open3(size, arr_list, board, stone)
#     list_ans = list(set(list_4).intersection(list_3_open))
#     return_list = []
#     for i in range(len(list_ans)):
#         if not is_banned_pos_new(board, list_ans[i], stone):  # 내 차례를 보는건지 상대 차례를 보는건지
#             return_list.append(list_ans[i])
#     return list_ans

# board상태에서 하나를 놨을 때 열린4 열린3이 만들어지는 자리들을 리턴
def get_next_43open(size,arr_list,board,stone):
    list_1 = get_next_open4(size, arr_list, board, stone)
    list_2 = get_next_open3(size,arr_list,board,stone)
    list_ans = list(set(list_1).intersection(list_2)) # 열린4 되는 좌표와 열린3 되는 좌표가 동일하다면 해당 좌표는 열린4와 열린3이 생성되는 자리
    return_list = []
    for i in range(len(list_ans)):
        if not is_banned_pos_new(board, list_ans[i], stone): # 해당 자리가 금수인지 판별
            return_list.append(list_ans[i])
    return return_list

# board상태에서 하나를 놨을 때 닫힌4 열린3이 만들어지는 자리들을 리턴
def get_next_43closed(size, arr_list, board, stone):
    list_1 = get_next_closed4(size, arr_list, board, stone)
    list_2 = get_next_open3(size, arr_list, board, stone)
    list_ans = list(set(list_1).intersection(list_2))  # 열린4 되는 좌표와 열린3 되는 좌표가 동일하다면 해당 좌표는 열린4와 열린3이 생성되는 자리
    return_list = []
    for i in range(len(list_ans)):
        if not is_banned_pos_new(board, list_ans[i], stone):  # 해당 자리가 금수인지 판별
            return_list.append(list_ans[i])
    return return_list


def get_next_33(size, arr_list, board, stone):
    list_3_open = get_next_open3(size, arr_list, board, stone)
    list_33 = []
    return_list = []

    for i in range(len(list_3_open)):
        if list_3_open.count(list_3_open[i]) >=2: # 좌표가 두개이상 중복되면 33
            list_33.append(list_3_open[i])

    for i in range(len(list_33)):
        if not is_banned_pos_new(board, list_33[i], stone):  # 내 차례를 보는건지 상대 차례를 보는건지
            return_list.append(list_33[i])
    return list(set(return_list)) # 중복 제거 후 리턴




def get_human_intervene_move(probs, board, black_white_ai,size=15):
    stone = get_stone_color(black_white_ai)
    enemy_stone = get_enemy_stone(stone)
    probs_tmp = copy.deepcopy(probs)  # 아마 ndarray(1,225)??
    # if self.is_test_mode:
    #     self.print_states_probs(probs_tmp)

    # 먼저, 놓자마자 바로 이길 수 있는 좌표가 있으면 해당 좌표를 선택하면 된다
    can_win_list = get_win_list(board,stone)  # 바로 이길 수 있는 위치 확인 (0~224 1차원 좌표)


    if _test_mode:
        print('------------------------------')
        print(f'AI : {black_white_ai}')
        print(f"이길 수 있는 좌표 리스트 : {can_win_list}")
    if len(can_win_list) >= 1:
        return random.choice(can_win_list)  # 어차피 리스트에 있는거 아무거나 놔도 이기므로 하나 랜덤으로 골라서 리턴

    # 이제 상대가 놓으면 바로 이길 수 있는 자리 탐색
    # can_lost_list나 can_win_list는 금수는 이미 처리하고 리턴됨
    can_lose_list = get_win_list(board, enemy_stone)  # 상대 입장에서 이기는 거 테스트 (type : (0~224 1차원 좌표))
    if _test_mode: print(f'질 수 있는 좌표 리스트 : {can_lose_list}')
    if len(can_lose_list) >= 1:  # 만약 존재한다면, 둘중에 probs가 높은 쪽을 막도록 작동
        arr_tmp = probs_tmp[0][can_lose_list]
        best_choice_idx = np.argmax(arr_tmp)
        best_move = can_lose_list[best_choice_idx]
        return best_move

    arr_list = board.states_loc

    can_attack_list_43_open = get_next_43open(size, arr_list, board, stone)  # 확실한 공격이 가능한 경우
    can_attack_list_43_closed = get_next_43closed(size, arr_list,board,stone) # 닫힌4 열린3

    can_attack_list_open4 = get_next_open4(size, arr_list, board, stone)  #
    can_attack_list_33 = []
    if _test_mode:
        print(f"공격 가능 43open : {can_attack_list_43_open}")
        print(f"공격 가능 4open : {can_attack_list_open4}")
    if stone == 2: # 열린 33공격은 백만 가능
        can_attack_list_33 = get_next_33(size, arr_list, board, stone)
        if _test_mode: print(f"공격 가능 33 : {can_attack_list_33}")

    can_attack_list_first = can_attack_list_43_closed + can_attack_list_43_open + can_attack_list_33+can_attack_list_open4# 열린4 , 열린43, 열린33처럼 상대가 선 공격할 수 있는 위치가 없다면 내가 승리가 확실한 수
    if len(can_attack_list_first) >= 1:
        arr_tmp = probs_tmp[0][can_attack_list_first]
        best_choice_idx = np.argmax(arr_tmp)
        best_move = can_attack_list_first[best_choice_idx]
        return best_move

    # 공격자 입장에서 탐색하면 (열린4)를 놓나 (열린4 닫힌3)을 놓나 (닫힌4 열린3)을 놓나 상대가 바로 이기는 상황이 아니면 똑같이 이길 수 밖에 없음
    # 하지만 방어하는 입장에서는 (열린4 열린3)이 될 수 있는 자리가 있는데 열린4만 될 수 있는 자리를 고르게 되면 패배함
    #   단순한 열린4의 경우 신경망 선에서 충분히 방어 가능?? 오히려 주도권이 있는 상황에서 열린4가 나올 수 있는 자리가 있다고 막으면 오히려 이길 기회를 놓칠 수 있음
    # can_defend_list_4 = get_next_open4(size, arr_list, board, stone)
    can_defend_list_43_open = get_next_43open(size, arr_list, board, enemy_stone)  # 상대가 나에게 확실한 공격이 가능한 경우
    can_defend_list_43_closed = get_next_43closed(size, arr_list, board, enemy_stone)  # 상대가 나에게 확실한 공격이 가능한 경우
    can_defend_list_33 = []
    if stone == 1: # 내가 흑이고 상대가 백이면 상대는 33 공격 가능
        can_defend_list_33 = get_next_33(size, arr_list, board, enemy_stone)

    if _test_mode:
        print(f"방어 필요 43open : {can_defend_list_43_open}")
        print(f"방어 필요 43closed : {can_defend_list_43_closed}")
        print(f"방어 필요 33 : {can_defend_list_33}")
        # print(f"방어 필요 4open : {can_defend_list_4}")


    # 열린 43을 우선 방어
    if len(can_defend_list_43_open) >= 1:
        arr_tmp = probs_tmp[0][can_defend_list_43_open]
        best_choice_idx = np.argmax(arr_tmp)
        best_move = can_defend_list_43_open[best_choice_idx]
        return best_move

    # 33이나 43(닫힌4)을 방어
    can_defend_list = can_defend_list_43_closed + can_defend_list_33
    if len(can_defend_list) >= 1:
        arr_tmp = probs_tmp[0][can_defend_list]
        best_choice_idx = np.argmax(arr_tmp)
        best_move = can_defend_list[best_choice_idx]
        return best_move

    return None # 특수 자리가 없는 경우

# 1차원 좌표를 2차원으로
# 시작 좌표 : (0,0)
# 15x15
def convert_xy_to_1nd(x, y, board_size):
    return board_size * y + x

# (y,x)꼴의 좌표를 1차원 move 좌표로 변경
def yx_to_move(size,location):
    if len(location) != 2: return error_const.CONST_WRONG_POSITION
    h, w = location[0], location[1]
    move = h * size + w
    if move not in range(size * size):
        return error_const.CONST_WRONG_POSITION
    return move

def convert_to_2nd_loc(size, index):  # 2차원 좌표로 변경
    y = index // size
    x = index - y * size
    return x, y


# 원하는 상태를 받아서 해당 상태의 금수 위치 리턴
def get_forbidden_new(board,stone_to_forbidden):
    board_size = board.width
    states_loc = board.states_loc
    forbidden_locations = None
    rule = Renju_Rule(states_loc, board_size)
    if stone_to_forbidden == 1:
        forbidden_locations, forbidden_types = rule.get_forbidden_points(stone=1)
    elif stone_to_forbidden == 2:  # 렌주룰에서 백은 금수 X
        forbidden_locations = []
        forbidden_types = []
    else:
        print(f"잘못된 stone 값 : {stone_to_forbidden}")
        quit()
    forbidden_moves = [yx_to_move(board_size,loc) for loc in forbidden_locations]
    return forbidden_moves

# 현재 상태에서 놓으면 바로 이길 수 있는 위치 찾기
def get_win_list(board, stone,size=15):
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
        if not is_banned_pos_new(board,win_list[i],stone): # 흑, 백이 다르게
            return_list.append(win_list[i])
    return return_list


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

