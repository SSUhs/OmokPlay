import numpy as np

from renju_rule import Renju_Rule

def get_all_stone_count(arr):
    return np.count_nonzero(arr)


def who_is_winner(arr,rule): # 15x15 형태의 numpy array
    if rule == 'renju':
        width = 15
        height = 15
        all_stone_count = get_all_stone_count(arr)
        arr_list = list(arr)
        if all_stone_count == width*height:
            return 0
        else:
            # rule = Renju_Rule(arr_list, width)
            for i in range(width):
                for j in range(height):
                    if arr_list[j][i] == 1 and is_contains_five(j,i, 1):
                        return 1
                    elif arr_list[i][j] == 2 and is_contains_five(j,i,2):
                        return 2
            # for문 다 돌았는데 없으면 >> 중간 기권으로 판정
            return 3
    else:
        print("아직 다른 룰은 누가 승자인지 판별 불가능")
        quit()
    return None

a = np.zeros([15,15])
a[1][2] = 1
a[2][3] = 1
a[3][4] = 1
a[4][5] = 1
a[5][6] = 1

print(who_is_winner(a,'renju'))
