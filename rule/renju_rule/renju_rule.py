from constant import error_const

empty = 0
# black_stone = 1
# white_stone = 2
# last_b_stone = 3
# last_a_stont = 4
# tie = 100

class Renju_Rule(object):
    def __init__(self, board, board_size):
        self.board = board
        self.board_size = board_size

    def is_invalid(self, x, y):
        return (x < 0 or x >= self.board_size or y < 0 or y >= self.board_size)

    def set_stone(self, x, y, stone):
        self.board[y][x] = stone

    def get_xy(self, direction):
        list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
        list_dy = [0, 0, -1, 1, -1, 1, -1, 1]
        return list_dx[direction], list_dy[direction]

    def get_stone_count(self, x, y, stone, direction):
        x1, y1 = x, y
        cnt = 1
        for i in range(2):
            dx, dy = self.get_xy(direction * 2 + i)
            x, y = x1, y1
            while True:
                x, y = x + dx, y + dy
                if self.is_invalid(x, y) or self.board[y][x] != stone:
                    break;
                else:
                    cnt += 1
        return cnt
    
    def is_gameover(self, x, y, stone):
        for i in range(4):
            cnt = self.get_stone_count(x, y, stone, i)
            if cnt >= 5:
                return True
        return False

    def is_six(self, x, y, stone):
        for i in range(4):
            cnt = self.get_stone_count(x, y, stone, i)
            if cnt > 5:
                return True
        return False

    def is_five(self, x, y, stone):
        for i in range(4):
            cnt = self.get_stone_count(x, y, stone, i)
            if cnt == 5:
                return True
        return False

    def find_empty_point(self, x, y, stone, direction):
        dx, dy = self.get_xy(direction)
        while True:
            x, y = x + dx, y + dy
            if self.is_invalid(x, y) or self.board[y][x] != stone:
                break
        if not self.is_invalid(x, y) and self.board[y][x] == empty:
            return x, y
        else:
            return None

    def open_three(self, x, y, stone, direction):
        for i in range(2):
            coord = self.find_empty_point(x, y, stone, direction * 2 + i)
            if coord:
                dx, dy = coord
                self.set_stone(dx, dy, stone)
                if 1 == self.open_four(dx, dy, stone, direction):
                    is_forbidden_point, err_code = self.forbidden_point(dx, dy, stone)
                    if not is_forbidden_point:
                        self.set_stone(dx, dy, empty)
                        return True
                self.set_stone(dx, dy, empty)
        return False

    def open_four(self, x, y, stone, direction):
        if self.is_five(x, y, stone):
            return False
        cnt = 0
        for i in range(2):
            coord = self.find_empty_point(x, y, stone, direction * 2 + i)
            if coord:
                if self.five(coord[0], coord[1], stone, direction):
                    cnt += 1
        if cnt == 2:
            if 4 == self.get_stone_count(x, y, stone, direction):
                cnt = 1
        else: cnt = 0
        return cnt

    def four(self, x, y, stone, direction):
        for i in range(2):
            coord = self.find_empty_point(x, y, stone, direction * 2 + i)
            if coord:
                if self.five(coord[0], coord[1], stone, direction):
                    return True
        return False

    def five(self, x, y, stone, direction):
        if 5 == self.get_stone_count(x, y, stone, direction):
            return True
        return False

    def double_three(self, x, y, stone):
        cnt = 0
        self.set_stone(x, y, stone)
        for i in range(4):
            if self.open_three(x, y, stone, i):
                cnt += 1
        self.set_stone(x, y, empty)
        if cnt >= 2:
            # print("double three")
            return True
        else: # 대각선 33을 못막는 오류가 있어서 수정
            self.set_stone(x,y,stone)
            if self.check_double3_new(x,y,stone):
                self.set_stone(x,y,empty)
                return True
            else:
                self.set_stone(x,y,empty)
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
    def get_stone_diff(self, arr_list, y, x, diff, dir):
        size = self.board_size
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


    def check_double3_new(self,x,y,stone):
        empty = 0
        is_left_right_3 = False # d
        is_up_down_3 = False  # a
        is_left_cross_3 = False  # b
        is_right_cross_3 =  False # c

        arr_list = self.board
        if self.get_stone_diff(arr_list,y,x,1,1) == stone and self.get_stone_diff(arr_list,y,x,1,-1) == stone \
            and self.get_stone_diff(arr_list,y,x,2,1) == empty and self.get_stone_diff(arr_list,y,x,2,-1) == empty:
            is_left_right_3 = True

        if self.get_stone_diff(arr_list,y,x,1,2) == stone and self.get_stone_diff(arr_list,y,x,1,-2) == stone \
            and self.get_stone_diff(arr_list,y,x,2,2) == empty and self.get_stone_diff(arr_list,y,x,2,-2) == empty:
            is_up_down_3 = True

        if self.get_stone_diff(arr_list,y,x,1,3) == stone and self.get_stone_diff(arr_list,y,x,1,-3) == stone \
            and self.get_stone_diff(arr_list,y,x,2,3) == empty and self.get_stone_diff(arr_list,y,x,2,-3) == empty:
            is_left_cross_3 = True

        if self.get_stone_diff(arr_list,y,x,1,4) == stone and self.get_stone_diff(arr_list,y,x,1,-4) == stone \
            and self.get_stone_diff(arr_list,y,x,2,4) == empty and self.get_stone_diff(arr_list,y,x,2,-4) == empty:
            is_right_cross_3 = True

        tf_list = [is_left_right_3,is_up_down_3,is_right_cross_3,is_left_cross_3]
        if tf_list.count(True) >= 3:
            return True
        else:
            return False


    def double_four(self, x, y, stone):
        cnt = 0
        self.set_stone(x, y, stone)
        for i in range(4):
            if self.open_four(x, y, stone, i) == 2:
                cnt += 2
            elif self.four(x, y, stone, i):
                cnt += 1
        self.set_stone(x, y, empty)
        if cnt >= 2:
            # print("double four")
            return True
        return False


    def forbidden_point(self, x, y, stone):
        if self.is_five(x, y, stone):
            return False, error_const.BANNED_OK
        elif self.is_six(x, y, stone): # 6목
            return True, error_const.BANNED_6
        elif self.double_three(x, y, stone):
            return True, error_const.BANNED_33
        elif self.double_four(x, y, stone):
            return True, error_const.BANNED_44
        return False, error_const.BANNED_OK  # 나머지 경우에는 문제 없음


    def get_forbidden_points(self, stone):
        coords = []
        forbidden_types = []  # ex : 3,3  4,4  6,6 중에 어떤건지
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                if self.board[y][x]: # 이미 돌이 놔져있는 곳은 스킵
                    continue
                is_forbidden_point, err_code = self.forbidden_point(x, y, stone)
                if is_forbidden_point:
                    coords.append((x, y))
                    forbidden_types.append(err_code)
        return [(y,x) for x,y in coords],forbidden_types
