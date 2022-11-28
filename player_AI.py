# train set을 사용하는 플레이어
import random

import numpy as np
from time import time
import tensorflow as tf
import keras.backend as K
import collections
import copy
import math


def convert_to_one_dimension(state):
    return np.concatenate(state)


def reshape_to_15_15_1(data):
    return K.reshape(data, [-1, 15, 15, 1])


# 1차원 좌표를 2차원으로
# 시작 좌표 : (0,0)
# 15x15
def convert_xy_to_1nd(x, y,board_size):
    return board_size * y + x

# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (좌우 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_right_5_can_win(arr_list, y, x, stone,reverse_dir=False):
    length = 0  # 이게 4가 되어야 5개중 4개가 놓인 상태이므로 이기는 것
    not_count = 0
    ans_x = None
    ans_y = None
    stone_reverse = None
    if stone == 1: stone_reverse = 2
    else : stone_reverse = 1
    gap = 6
    add = 1
    if reverse_dir:
        gap = -6
        add = -1

    for s in range(x, x + gap,add): # 6목도 고려해서 6까지
        if s >= 15 or s < 0: # 좌표 초과
            break
        if arr_list[y][s] == stone:
            length +=1
            continue
        elif arr_list[y][s] == stone_reverse: # 상대 돌 만나면 종료
            not_count += 1
            break
        else: # 놓아야 할 자리
            if not_count == 0:
                ans_x = s
                ans_y = y
            not_count +=1
    if length == 4 and not_count <= 2 and (ans_x is not None):
        return True, ans_x, ans_y
    elif length == 5 and stone == 2 and not_count == 1 and (ans_x is not None):  # 백이면서 6개중에 1개만 비어있고, 하나 놓으면 6목이 되는 경우
        return True, ans_x, ans_y
    else:
        return False,None,None

def check_left_5_can_win(arr_list, y, x, stone):
    return check_right_5_can_win(arr_list, y, x, stone,reverse_dir=True)

# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (아래 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_down_5_can_win(arr_list, y, x, stone,reverse_dir=False):
    length = 0
    not_count = 0
    ans_x = None
    ans_y = None
    end_for = y+6
    stone_reverse = None
    if stone == 1: stone_reverse = 2
    else : stone_reverse = 1
    add = 1
    if reverse_dir:
        end_for = y-6
        add = -1
    for s in range(y, end_for,add):
        if s >= 15 or s < 0:
            break
        if arr_list[s][x] == stone:
            length += 1
            continue
        elif arr_list[s][x] == stone_reverse: # 상대 돌 만나면 종료
            not_count += 1
            break
        else:
            if not_count == 0:
                ans_x = x
                ans_y = s
            not_count += 1
    if length == 4 and not_count <= 2 and (ans_x is not None): # 흑 or 백인 상황에서 5개중 1개만 비어있는 경우 하나 놓으면 이김
        return True,ans_x,ans_y
    elif length == 5 and stone == 2 and not_count ==1 and (ans_x is not None): # 백이면서 6개중에 1개만 비어있고, 하나 놓으면 6목이 되는 경우
        return True,ans_x,ans_y
    else:
        return False,None,None

# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (대각선 오른쪽 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_down_cross_right_5_can_win(arr_list, y, x, stone,reverse_dir=False):
    length = 0
    not_count = 0
    ans_x = None
    ans_y = None
    stone_reverse = None
    if stone == 1: stone_reverse = 2
    else : stone_reverse = 1
    for i in range(6):
        if reverse_dir: i = - i
        if y + i >= 15 or x + i >= 15 or x + i < 0 or y + i < 0: # 좌표 초과
            break
        elif arr_list[y + i][x + i] == stone:
            length += 1
        elif arr_list[y + i][x + i] == stone_reverse:
            not_count += 1
            break
        else: # 돌이 없을 때 : 해당 위치가 놓을 자리
            if not_count == 0:
                ans_x = x + i
                ans_y = y + i
            not_count+=1
        if reverse_dir: i = - i
    if length == 4 and not_count <= 2 and (ans_x is not None):
        return True,ans_x,ans_y
    elif length == 5 and stone == 2 and not_count == 1 and (ans_x is not None):
        return True, ans_x, ans_y
    else:
        return False,None,None

# arr_list : list(ndarray(15,15)) 형태
# 하나 더 놓으면 이길 수 있는 위치 찾기 (대각선 왼쪽 방향)
# stone : 1이면 흑 white면 2
# return : (있는지 + 있다면 x y좌표)
def check_down_cross_left_5_can_win(arr_list, y, x, stone,reverse_dir=False):
    length = 0
    not_count = 0
    ans_x = None
    ans_y = None
    stone_reverse = None
    if stone == 1: stone_reverse = 2
    else : stone_reverse = 1
    for i in range(6):
        if reverse_dir: i = -i
        if y + i < 0 or y - i < 0 or y + i >= 15 or x - i >= 15: # 좌표 초과
            break
        elif arr_list[y + i][x - i] == stone:
            length += 1
        elif arr_list[y + i][x - i] == stone_reverse:
            not_count += 1
            break
        else: # 돌이 없을 때 : 해당 위치가 놓을 자리
            if not_count == 0:
                ans_x = x - i
                ans_y = y + i
            not_count += 1
        if reverse_dir : i = -i
    if length == 4 and not_count <= 2 and (ans_x is not None):
        return True,ans_x,ans_y
    elif length == 5 and stone == 2 and not_count == 1 and (ans_x is not None):
        return True,ans_x,ans_y
    else:
        return False,None,None


def check_up_cross_right_5_can_win(arr_list, y, x, stone):
    return check_down_cross_left_5_can_win(arr_list, y, x, stone,reverse_dir=True)


def check_up_cross_left_5_can_win(arr_list, y, x, stone):
    return check_down_cross_right_5_can_win(arr_list, y, x, stone,reverse_dir=True)


def check_up_5_can_win(arr_list, y, x, stone):
    return check_down_5_can_win(arr_list, y, x, stone,reverse_dir=True)



def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs



class player_AI():
    def __init__(self, size, is_test_mode, black_white_human, train_num, is_sequential_model=True, use_mcts_search=False,is_self_play=False,is_human_intervene=False):
        self.size = size
        self.is_self_play = is_self_play
        self.is_test_mode = is_test_mode
        self.black_white_human = black_white_human # 참고로 사람이 흑을 하면 AI는 백을 로딩해야됨
        self.black_white_ai = None
        if black_white_human == 'black':
            self.black_white_ai = 'white'
        else:
            self.black_white_ai = 'black'

        self.model = self.load_model(model_type='policy',black_white_ai=self.black_white_ai, train_num=train_num)
        self.is_sequential_model = is_sequential_model
        self.value_net_model = self.load_model(model_type='value', black_white_ai=self.black_white_ai,
                                               train_num=train_num)
        self.is_human_intervene = is_human_intervene  # 사람 알고리즘 개입 (ex : 닫힌 4 무조건 막기 )
        self.use_mcts_search = use_mcts_search  # MCTS 검색을 쓸 것인지 아니면 단순히 가장 probs가 높은 걸로 리턴할 것인지
        if self.use_mcts_search:
            self.mcts = MCTS_TrainSet(self.model, c_puct=5, n_playout=400, is_test_mode=is_test_mode, board_size=size, value_net_model=self.value_net_model)
            # value_net_tmp = ValueNetTmpNumpy(board_size=size,net_params_file=f'tf_value_{size}_{train_num}_{self.black_white_ai}.pickle')# numpy로 임시로 구현한 가치망

    def convert_to_2nd_loc(self, index):  # 2차원 좌표로 변경
        y = index // self.size
        x = index - y * self.size
        return x,y

    def load_model(self, model_type,black_white_ai, train_num):
        model_file = None
        if model_type == 'policy':
            model_file = f'./model_train/tf_policy_{self.size}_{train_num}_{black_white_ai}.h5'
            model = tf.keras.models.load_model(model_file)
        elif model_type == 'value':
            model_file = f'./model_train/tf_value_{self.size}_{train_num}_{black_white_ai}.h5'
            model = tf.keras.models.load_model(model_file,compile=False)
        else:
            print("잘못된 타입")
            quit()

        return model

    # n*n 형태를 일차원으로
    def get_action(self, board):
        # state : numpy
        state = board.get_states_by_numpy()
        if self.is_sequential_model:
            inputs = reshape_to_15_15_1(state)  # 현재 상태. 이 상태를 기반으로 예측
            if self.use_mcts_search:
                move,value = self.get_move_mcts(board, inputs)  # mcts를 사용해서 추가 예측
            else:
                move,value = self.get_move_not_mcts(board, inputs)  # mcts 없이 단순히 확률이 가장 높은 경우를 선택
            x, y = self.convert_to_2nd_loc(move)
            print(f"선택 좌표 (0,0부터) : {move} = ({x},{y})")
            print(f'가치망 value : {value}')
            return move
        else:
            print("sequential 아닌 것은 아직 구현 X")
            quit()




    # 현재 상태에서 놓으면 바로 이길 수 있는 위치 찾기
    # my_turn : True면 현재 상태에서 내가 놓을 거, False면 상대의 경우로 따지는 것
    #           (my_turn을 False로 하면, 상대 입장에서 놓으면 이길 수 있는 리스트를 획득 가능)
    def get_win_list(self,board,my_turn):
        is_black = board.is_you_black()
        stone = None # 흑이면 1 백이면 2
        if my_turn: # 내 차례인데 is_black >> "흑"
            if is_black:
                stone = 1
            else:
                stone = 2
        else: # 내 차례가 아닌데 예측하는 경우
            if is_black:
                stone = 2
            else:
                stone = 1

        # arr_list : ndarray(15,15) 형태를 list()로
        arr_list = board.states_loc
        size = self.size # 15x15면 15

        win_list = [] # 0~224의 좌표
        for y in range(size):
            for x in range(size):
                if arr_list[y][x] == stone:
                    tf,x_new,y_new = check_right_5_can_win(arr_list, y, x, stone)
                    if tf: win_list.append(convert_xy_to_1nd(x_new,y_new,size))

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
            if not self.is_banned_pos(board,win_list[i],my_turn): # 내 차례를 보는건지 상대 차례를 보는건지
                return_list.append(win_list[i])
        return return_list


    # 금수 or 이미 수가 놓아지지 않은 자리 중에서 가장 최선의 인덱스
    def get_best_idx(self, probs, board):
        probs_tmp = copy.deepcopy(probs) # 아마 ndarray(1,225)??
        if self.is_test_mode:
            print("정책망 확률 표 (0,0 부터 시작)")
            self.print_states_probs(probs_tmp)

        # 특수상황에서 사람의 알고리즘 강제 개입 하는 경우
        if self.is_human_intervene:
            # 먼저, 놓자마자 바로 이길 수 있는 좌표가 있으면 해당 좌표를 선택하면 된다
            can_win_list = self.get_win_list(board,True) # 바로 이길 수 있는 위치 확인 (0~224 1차원 좌표)
            print(f"이길 수 있는 좌표 리스트 : {can_win_list}")
            if len(can_win_list) >= 1:
                return random.choice(can_win_list) # 어차피 리스트에 있는거 아무거나 놔도 이기므로 하나 랜덤으로 골라서 리턴

            # 이제 상대가 놓으면 바로 이길 수 있는 자리 탐색
            # can_lost_list나 can_win_list는 금수는 이미 처리하고 리턴됨
            can_lose_list = self.get_win_list(board,False) # 상대 입장에서 이기는 거 테스트 (type : (0~224 1차원 좌표))
            print(f'질 수 있는 좌표 리스트 : {can_lose_list}')
            if len(can_lose_list) >= 1: # 만약 존재한다면, 둘중에 probs가 높은 쪽을 막도록 작동
                arr_tmp = probs_tmp[0][can_lose_list]
                best_choice_idx = np.argmax(arr_tmp)
                best_move = can_lose_list[best_choice_idx]
                return best_move
            
            # code20221127201449
        
        # 인간이 개입 했는데도 못찾았으면 최선을 찾는다
        while True:
            best_index = np.argmax(probs_tmp[0])
            # 이미 돌이 있는 자리를 선택하거나 금수에 놓은 경우
            if self.is_banned_pos(board, best_index):
                probs_tmp[0][best_index] = -1  # 금수 자리는 선택 불가능 하게 설정
                continue
            else:
                break
        return best_index


    # (디버그 용도) 확률 리턴
    # ndarray_probs_1nd : ndarray(1,225) # 15x15 기준
    def print_states_probs(self,ndarray_probs_1nd_):
        length = self.size*self.size
        ndarray_probs_1nd = copy.deepcopy(ndarray_probs_1nd_)
        list_print = []
        for i in range(length):
            best_idx = np.argmax(ndarray_probs_1nd[0])
            best_prob_float = ndarray_probs_1nd[0][best_idx]
            ndarray_probs_1nd[0][best_idx] = -1  # 그 다음 최대를 찾기 위해 -1로 수정
            if best_prob_float <= 0.001:
                continue
            x,y = self.convert_to_2nd_loc(best_idx)
            list_print.append(f'({x},{y}) : {format(best_prob_float,".4f")}%')
        for i in range(len(list_print)):
            print(list_print[i])

    def get_move_not_mcts(self, board, input):
        probs = self.model.predict(input)
        value = self.value_net_model.predict(input)
        return self.get_best_idx(probs, board),value

    # 이미 수가 놓아진 자리 or 금수 자리
    # index : 0~224의 1차원화 좌표
    # my_turn : 내 차례가 아니라 상대 차례를 예측 하는 경우, 흑백을 뒤집어야함
    def is_banned_pos(self, board, index,my_turn=True):
        is_black = None
        if my_turn:
            is_black = board.is_you_black()
        else:
            is_black = not board.is_you_black()
        if (index in board.states) or (is_black and (index in board.forbidden_moves)):
            return True
        else:
            return False

    # MCTS 기반
    def get_move_mcts(self, board, input):
        # np.zeros : 0으로만 채워진 배열 생성하는 함수
        if board.width * board.height - len(board.states) > 0:  # 보드판이 꽉 안찬 경우
            # acts와 probs에 의해 착수 위치가 정해진다.
            time_get_probs = time()  # probs를 얻는데까지 걸리는 시간
            probs = self.mcts.get_move_probs(board)
            # probs = self.model.predict(input)  # 위치별 확률
            if self.is_test_mode:
                time_gap = time() - time_get_probs
                print(f'get_probs 하는데 소요된 시간 : {time_gap}')
            if self.is_self_play:
                # (자가 학습을 할 때는) Dirichlet 노이즈를 추가하여 탐색
                print("강화 학습은 구현중")
                # move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                # time_update_with_move = time()
                # self.mcts.update_with_move(move)
                # if self.is_test_mode:
                #     print(f'update_with_move 하는데 소요된 시간 : {time() - time_update_with_move}')
            else:  # 플레이어와 대결하는 경우
                move = self.get_best_idx(probs, board)  # 금수 or 이미 놓지 않은 자리중에서 가장 좋은 자리 선택
                print("mcts ai가 고른 자리 : ", move)
                # 점검
                self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board_img is full")

    def set_player_ind(self, p):
        self.player = p


class MCTS_TrainSet(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, model, c_puct=5, n_playout=400, is_test_mode=False, board_size=None,value_net_model=None):
        """
        policy_value_fn: a function that takes in a board_img state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self.board_size = board_size
        self._root = TreeNode(None, DummyNode(), board_size)
        self._policy_model = model
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.is_test_mode = is_test_mode
        self.value_net_model = value_net_model

    def get_zero_board(self):
        zero_board = np.zeros(self.board_size * self.board_size)
        return zero_board

    # 임시 가치망
    def make_value_tmp(self,state):
        value = self.value_net.value_fn(state)
        return value

    # num_reads = _n_playout??
    def UCT_search(self, state_b, num_reads):
        beforeTime = time()
        self._root = TreeNode(None, DummyNode(), self.board_size)
        for _ in range(num_reads):
            state = copy.deepcopy(state_b)
            root = self._root
            leaf = root.select_leaf(state)  # leaf : 노드 객체
            # child_priors가 결국 (82,)가 되든 (81,)가 되든 해야됨
            child_priors = self._policy_model.predict(state)  # NeuralNet.evaluate(leaf.game_state)
            print(f'child_priors shape : {child_priors.shape}')
            value_estimate = self.value_net_model.predict(state)
            end, winner = state.game_end()
            if end:  # 누군가 이기거나 draw
                # for end state，return the "true" leaf_value
                # winner은 무승부의 경우 -1이고, 우승자가 존재하면 우승자 int (0,1이였나 1,2였나)
                if winner == -1:  # tie (무승부)
                    value_estimate = 0.0  # 무승부의 경우 leaf_value를 0으로 조정 (value_estimate = leaf_value)
                else:
                    value_estimate = (
                        1.0 if winner == state.get_current_player() else -1.0)  # 우승자가 자신이라면, leaf_value는 1로, 패배자라면 -1로
                leaf.backup(value_estimate)
                continue  # continue한다는건 한판 더 한다는 것
            else:  # 게임에서 못이긴 경우
                leaf.expand(child_priors)
                leaf.backup(value_estimate)
        if self.is_test_mode:
            currentTime = time()
            print(f'UCT-searchTime : {currentTime - beforeTime}')
            beforeTime = currentTime

    # 여기서 state는 game.py의 board 객체
    def get_move_probs(self, state, temp=1e-3):
        # 이 for 문은 AI가 돌리는 for문
        # _n_playout 횟수가 찰 때까지 playout 수행
        # 따라서, _n_playout가 높을 수록 수행 횟수가 많아 지므로, 소요 시간이 늘어나고 성능은 늘어남
        # 학습할 때 자가대전 몇판 하냐랑은 다른 것
        # n_playout이 400이면, 400번 수행해서 나온 가중치들을 기준으로 확률 배열 리턴

        self.UCT_search(state, self._n_playout)
        # acts = 위치번호 / visits = 방문횟수
        # acts = self._root.child_number_visits[]
        # visits = self._root.child_number_visits
        # list로 바꿔야함!!
        # 원본 : visits는 tuple이다
        acts_list = []
        visits_list = []
        cur_dict_key = state.states.keys()
        for i in range(self.board_size * self.board_size):
            if not (i in cur_dict_key):  # 돌이 안놓인 위치의 경우 추가
                acts_list.append(i)
                visits_list.append(int(self._root.child_number_visits[i]))

        acts = tuple(acts_list)
        visits = tuple(visits_list)
        # print(f'acts_list : {acts} / size {len(acts)}')
        # print(f'visits_list : {visits} / size {len(visits)}')
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    # 플레이어 대결의 경우 컴퓨터는 update_with_move를 호출 할 때 last_move 파라미터를 -1로 전달 (아직 무슨 의미인지는 파악 X)
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]  # 돌을 둔 위치가 root노드가 됨
            self._root._parent = None
        else:
            self._root = TreeNode(None, DummyNode(), self.board_size)

    def __str__(self):
        return "MCTS"


class TreeNode(object):
    """ MCTS 트리의 노드.
    Q : its own value
    P : prior probability
    u : visit-count-adjusted prior score
    """

    def __init__(self, move, parent, board_size):
        # self.game_state = game_state
        self.is_expanded = False
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        # self.number_visits = 0 있어야하는데 numpy 방식에서는 제거
        # self._Q = 0
        self._u = 0
        # self._P = prior_p
        self.move = move
        self.board_size = board_size
        np_size = (board_size * board_size)  # ex : 15*15면 226?? 225??. 디폴트 바둑 형태의 경우 이 값에 362가 들어갔음
        self.child_priors = np.zeros([np_size], dtype=np.float32)
        self.child_total_value = np.zeros([np_size], dtype=np.float32)
        self.child_number_visits = np.zeros([np_size], dtype=np.float32)

    def expand(self, child_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability according to the policy function.
        """
        # action : int 타입
        self.is_expanded = True
        # print("expand 전 shpae : ",self.child_priors.shape) # (82,)
        self.child_priors = child_priors
        # print("expand 후 shpae : ",self.child_priors.shape)  # (1, 81)

    def select_leaf(self, state):
        current = self
        # print(f'select_leaf type : {type(current)}')
        while current.is_expanded:
            # print("노드 타입 : ",type(current))
            current.number_visits += 1  # Optimizing for performance using NumPy
            current.total_value -= 1  # Optimizing for performance using NumPy
            best_move = current.best_child()
            current = current.maybe_add_child(best_move, state.forbidden_moves, state.is_you_black())
            state.do_move(best_move)  # 리프노드가 나올 때까지 move
        return current

    # def add_child(self, move, prior):  # move = action (아마 int)
    #     self._children[move] = TreeNode(move, parent=self)

    # maybe가 붙는 이유가, move(action)이 self._children 안에 없는 경우에만 적용되기 떄문인듯
    def maybe_add_child(self, move, forbidden_moves, is_you_black):
        if move not in self._children:
            # print("move는 action이므로 type이 int가 나와야함. 그리고 0~225사이값")
            # print(f'move type : {type(move)} / move값 : {move}')
            if is_you_black and move in forbidden_moves:  # 흑돌일 때 금수 위치는 확장노드에 집어 넣지 않음
                return self
            else:
                self._children[move] = TreeNode(move, parent=self, board_size=self.board_size)
        return self._children[move]

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        # leaf node : 자식 노드가 없는 노드
        return self._children == {}

    def is_root(self):
        return self._parent is None

    @property
    def number_visits(self):  # 주의!! DummyNode가 이 함수를 실행시 오류 발생 (DummyNode는 Parent가 없기 때문)
        return self._parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self._parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self._parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self._parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        # print(type(self.number_visits))
        # print(type(self.child_priors))
        # print(type(self.child_number_visits))
        # print(f'shpae child_priors : {self.child_priors.shape}')  # (1, 81)
        # print(f'shpae child_number_visits : {self.child_number_visits.shape}')  #  (82,)
        return math.sqrt(self.number_visits) * (self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def backup(self, value_estimate):  # 루트노드 = 부모None >> 루트노드까지 계속 반복
        current = self
        while current._parent is not None:
            current.total_value += value_estimate + 1
            current = current._parent


class DummyNode(object):
    def __init__(self):
        self._parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
