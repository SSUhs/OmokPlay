import numpy as np


# ex) 0~224 형태의 보드판을 15*15 형태로 변경
def convert_1nd_board_to_2nd(arr_1nd,board_size):
    return np.reshape(arr_1nd,(board_size,board_size))


def convert_2nd_board_to_1nd(arr_2nd):
    return np.ravel(arr_2nd)

# 0도(원본),90도,180도,270도 회전 + 각각마다 상하,좌우 반전 >> 하나의 상태로 12가지 상태 추출
def get_rotate_board_12dir(arr):
    all_list = []
    for i in range(4):
        rotates = np.rot90(arr,i)
        flip_horizontal = np.fliplr(rotates)
        flip_vertical = np.flipud(rotates)
        all_list.append(rotates)
        all_list.append(flip_horizontal)
        all_list.append(flip_vertical)
    return all_list

def convert_to_label_start0(x, y,board_size):
    return board_size * y + x

def get_rotate_label(label,board_size):
    arr = convert_label_to_board(label,board_size=board_size)
    rotate_list = get_rotate_board_12dir(arr) # len = 12
    all_list = []
    for i in range(len(rotate_list)):
        label_arr = rotate_list[i]
        y,x = np.where(label_arr == 1)
        move = convert_to_label_start0(x[0],y[0],board_size) # ndarray 형태로 나오기 때문에 0번으로 접근
        all_list.append(move)
    return all_list


# move : 15*15의 경우 0~224
# 레이블을 board형태로 전환
def convert_label_to_board(move,board_size):
    y = move//board_size
    x = move-(y*board_size)
    tmp = np.zeros([board_size,board_size])
    tmp[y][x] = 1
    return tmp




test_1nd = np.linspace(0, 224, 225)
arr_2nd = convert_1nd_board_to_2nd(test_1nd,15)
arr_1nd = convert_2nd_board_to_1nd(arr_2nd)
arr = np.zeros([5,5])
arr[1][2] = 1
arr[2][3] = 2
board_size = 5
print(get_rotate_label(13,board_size))

