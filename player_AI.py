# train set을 사용하는 플레이어
from time import time
import tensorflow as tf
import keras.backend as K
from rule.renju_rule.renju_helper import *


def load_model_trainset_mode(model_type, size, train_num):
    model = None
    model_file = None
    if model_type == 'policy':
        model_file = f'./model_train/tf_policy_{size}.h5'  # 현재 흑백 통합
        model = tf.keras.models.load_model(model_file)
    elif model_type == 'value':
        model_file = f'./model_train/tf_value_{size}.h5'  # 현재 흑백 통합
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





def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


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

    def expand(self, action_priors, forbidden_moves, is_you_black,board):
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
        for action, prob in action_priors:
            if is_you_black and action in forbidden_moves:
                continue
            if action not in self._children:  #  code20221130141219 #
                if prob < 0.1: # 확률이 너무 낮은 부분은 확장하지 않음
                    continue
                if action in board.states:
                    continue
                self._children[action] = TreeNode(self, prob,self.depth+1)

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
    # 부모 노드로 갈 때마다 -1 곱해주는 이유 : 흑 돌의 부모 노드는 백돌이고 백돌의 부모 노드는 흑돌 (승 패가 뒤집힌다)
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
        policy_value_fn: a function that takes in a board_img_15 state and outputs
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
    # stone : play_out을 수행하는 돌 색깔
    def _playout(self, state, stone,black_white_ai):
        node = self._root
        while (1):
            # 리프 노드가 나올 때까지 계속 진행
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            stone_tmp = 1 if state.is_you_black() else 2
            state.do_move(action,stone_tmp)  # 리프노드가 나올 때 까지 move
        set_all = set(range(state.width * state.height))
        set_state_keys = set(state.states.keys())
        legal_positions = list(set_all - set_state_keys)
        np_states = state.get_states_by_numpy()
        inputs = reshape_to_15_15_1(np_states)  # 현재 상태. 이 상태를 기반으로 예측
        act_probs = self.policy_net.predict(inputs,verbose=0)
        leaf_value = self.value_net.predict(inputs,verbose=0)[0][0]
        if black_white_ai == 'white': # 가치망은 흑을 기준으로 나와 있으므로 뒤집어야함
            leaf_value = -leaf_value
        legal_arr = act_probs[0][legal_positions]  # 얘는 수를 놓을 때마다 사이즈가 줄어 듦  # 왜 0번이냐면 애초에 act_probs가 [1][225] 이런형태라 그럼
        action_probs = zip(legal_positions, legal_arr)
        # end (bool 타입) : 게임이 단순히 끝났는지 안끝났는지 (승,패 또는 화면 꽉찬 경우에도 end = True)
        end, winner_stone = state.game_end()
        if not end: # asdf
            forbidden_moves = get_forbidden_new(state,1) # 현재 노드 상태에서의 금수 위치
            node.expand(action_probs, forbidden_moves, state.is_you_black(),state)
        else:
            # for end state，return the "true" leaf_value
            # winner은 무승부의 경우 -1이고, 우승자가 존재하면 우승자 int (0,1이였나 1,2였나)
            if winner_stone == -1:  # tie (무승부)
                leaf_value = 0.0  # 무승부의 경우 leaf_value를 0으로 조정
            else:
                leaf_value = (1.0 if winner_stone == stone else -1.0)  # 우승자가 자신이라면, leaf_value는 1로, 패배자라면 -1로
        node.update_recursive(-leaf_value)

    # 여기서 state는 game.py의 board 객체
    def get_move_probs(self, state, stone, black_white_ai, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy,stone, black_white_ai)
            # print(f"playout : {n}번 수행")

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
            move, value = self.get_move_not_mcts(board, inputs,black_white_ai)  # mcts 없이 단순히 확률이 가장 높은 경우를 선택
        x, y = convert_to_2nd_loc(board.width, move)
        if self.is_test_mode:
            print(f"선택 좌표 (0,0부터) : {move} = ({x},{y})")
            print(f'가치망 value : {value}')
        return move



    def get_move_not_mcts(self, board, input,black_white_ai):
        # print(f"code20221207145614 : {type(input)} / {len(input)}")
        probs = self.policy_net.predict(input)
        value = self.value_net.predict(input,steps=1)
        size = board.width

        stone = get_stone_color(black_white_ai)
        enemy_stone = get_enemy_stone(stone)

        if self.is_human_intervene:
            move_intervene = get_human_intervene_move(probs, board,black_white_ai,size=board.width)
            if move_intervene is not None:
                if stone == 1 and move_intervene in board.forbidden_moves:
                    print("오류 : 금수가 제거되지 않은 상태로 move리턴됨")
                    quit()
                if self.is_test_mode:
                    print(f"방어 및 공격 로직에 따라 결정 : {convert_to_2nd_loc(size,move_intervene)}")
                return move_intervene,value

        # 특수 알고리즘에 해당 안되면 최대 확률 부분을 찾는다

        while True:
            best_index = np.argmax(probs[0])
            # 이미 돌이 있는 자리를 선택하거나 금수에 놓은 경우
            if is_banned_pos_new(board, best_index, stone):
                probs[0][best_index] = -1  # 금수 자리는 선택 불가능 하게 설정
                continue
            elif self.is_human_intervene: # 만약 내가 지금 놓을려는 자리가 다음에 내가 바로 4를 만드는 경우가 아닌데 열린 3이 존재하면 그때는 열린3을 막는방향으로
                arr_list = board.states_loc
                x_best,y_best = convert_to_2nd_loc(size,best_index)
                is_my_open4 = is_4_when_do(size,arr_list,stone,x_best,y_best) # best_index자리에 놨을 때 내가 주도권 가지는지
                if not is_my_open4: # 만약 내가 주도권을 가질 상황이 아닌데 상대한테 현재 열린 3이 존재하면
                    enemy_open_3_list = get_next_open4(size,arr_list,board,enemy_stone)  # 열린 3은 하나 놓으면 열린4
                    if len(enemy_open_3_list) >= 1: # 존재한다면 이건 강제로 막지 않으면 패배
                        if self.is_test_mode: print(f"AI가 원래 선택한 자리 : ({x_best},{y_best})")
                        while True:
                            arr_tmp = probs[0][enemy_open_3_list] # 여러개라면 확률이 제일 높은 걸 고른다
                            new_best_choice = np.argmax(arr_tmp)
                            new_best_move = enemy_open_3_list[new_best_choice]
                            if probs[0][new_best_move] < -0.99: # 모든 방어 자리가 나의 금수인 경우, 어차피 지는 상황이므로 원래 고른거 반환
                                return best_index,value
                            elif is_banned_pos_new(board, new_best_move, stone): # 혹시나 방어해야되는 자리가 금수 자리인 경우
                                probs[0][new_best_move] = -1
                                continue
                            else:
                                break
                        return new_best_move, value
                    else: # 상대 열린3 없으면 넘김
                        break
                else:
                    break
            else:
                break
        return best_index, value

    # MCTS 기반
    def get_move_mcts(self, board, black_white_ai):
        # np.zeros : 0으로만 채워진 배열 생성하는 함수
        size = board.width
        stone = get_stone_color(black_white_ai)
        enemy_stone = get_enemy_stone(stone)
        if board.width * board.height - len(board.states) > 0:  # 보드판이 꽉 안찬 경우
            move = None
            inputs = reshape_to_15_15_1(board.get_states_by_numpy())  # 현재 상태. 이 상태를 기반으로 예측
            value_current = self.value_net.predict(inputs)[0]
            if black_white_ai == 'white':  # 백이면 가치망 뒤집는다
                value_current = -value_current
                
            # 인간의 알고리즘 개입
            if self.is_human_intervene:
                # 먼저, 놓자마자 바로 이길 수 있는 좌표가 있으면 해당 좌표를 선택하면 된다
                probs_tmp = self.policy_net.predict(inputs)
                move_intervene = get_human_intervene_move(probs_tmp,board,black_white_ai,size)
                if move_intervene is not None:
                    return move_intervene,value_current

            # 사람이 개입할 자리가 없거나 사람 개입이 False인 경우
            # acts와 probs에 의해 착수 위치가 정해진다.
            time_get_probs = time()  # probs를 얻는데까지 걸리는 시간
            acts, probs = self.mcts.get_move_probs(board,stone, black_white_ai)
            if self.is_test_mode:
                time_gap = time() - time_get_probs
                print(f'get_probs 하는데 소요된 시간 : {time_gap}')
            if self._is_selfplay:
                # (자가 학습을 할 때는) Dirichlet 노이즈를 추가하여 탐색
                print("강화 학습은 구현 X")
                # move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                # time_update_with_move = time()
                # self.mcts.update_with_move(move)
                # if self.is_test_mode:
                #     print(f'update_with_move 하는데 소요된 시간 : {time() - time_update_with_move}')
            else:  # 플레이어와 대결하는 경우
                move = np.random.choice(acts, p=probs)
                if self.is_human_intervene:  # 만약 내가 지금 놓을려는 자리가 다음에 내가 바로 4를 만드는 경우가 아닌데 열린 3이 존재하면 그때는 열린3을 막는방향으로
                    arr_list = board.states_loc
                    x_best, y_best = convert_to_2nd_loc(size, move)
                    is_my_open4 = is_4_when_do(size, arr_list, stone, x_best, y_best)  # best_index자리에 놨을 때 내가 주도권 가지는지
                    if not is_my_open4:  # 만약 내가 주도권을 가질 상황이 아닌데 상대한테 현재 열린 3이 존재하면
                        enemy_open_3_list = get_next_open4(size, arr_list, board, enemy_stone)  # 열린 3은 하나 놓으면 열린4
                        if len(enemy_open_3_list) >= 1:  # 존재한다면 이건 강제로 막지 않으면 패배
                            probs_new = self.policy_net.predict(inputs) # 확률 배열
                            while True:
                                arr_tmp = probs_new[0][enemy_open_3_list]  # 여러개라면 확률이 제일 높은 걸 고른다
                                new_best_choice = np.argmax(arr_tmp)
                                new_best_move = enemy_open_3_list[new_best_choice]
                                if probs_new[0][new_best_move] < -0.99:  # 모든 방어 자리가 나의 금수인 경우, 어차피 지는 상황이므로 원래 고른거 반환
                                    return move, value_current
                                if is_banned_pos_new(board, new_best_move, stone):  # 혹시나 방어해야되는 자리가 금수 자리인 경우
                                    probs_new[0][new_best_move] = -1 # -1로 조정
                                    continue
                                else:
                                    break
                            return new_best_move, value_current
                print("mcts ai가 고른 자리 : ", move)
                # 점검
                self.mcts.update_with_move(-1)
            return move, value_current
        else:
            print("WARNING: the board_img_15 is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
