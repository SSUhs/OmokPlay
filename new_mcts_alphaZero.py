import math

import numpy as np
import copy
import collections
from time import time


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """ MCTS 트리의 노드.
    Q : its own value
    P : prior probability
    u : visit-count-adjusted prior score
    """

    def __init__(self, move, parent,board_size):
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
        np_size = (board_size*board_size)  # ex : 15*15면 226?? 225??. 디폴트 바둑 형태의 경우 이 값에 362가 들어갔음
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
            current = current.maybe_add_child(best_move,state.forbidden_moves, state.is_you_black())
            state.do_move(best_move)  # 리프노드가 나올 때까지 move
        return current

    # def add_child(self, move, prior):  # move = action (아마 int)
    #     self._children[move] = TreeNode(move, parent=self)

    # maybe가 붙는 이유가, move(action)이 self._children 안에 없는 경우에만 적용되기 떄문인듯
    def maybe_add_child(self, move,forbidden_moves,is_you_black):
        if move not in self._children:
            # print("move는 action이므로 type이 int가 나와야함. 그리고 0~225사이값")
            # print(f'move type : {type(move)} / move값 : {move}')
            if is_you_black and move in forbidden_moves: # 흑돌일 때 금수 위치는 확장노드에 집어 넣지 않음
                return self
            else:
                self._children[move] = TreeNode(move,parent=self,board_size=self.board_size)
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
        # print(f'type : 더미노드 child_total_value: {type(self.child_total_value)}')
        # print(f'type : 더미노드 child_number_visits: {type(self.child_number_visits)}')

class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, is_test_mode=False, board_size=None):
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
        self._root = TreeNode(None, DummyNode(),board_size)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.is_test_mode = is_test_mode



    def get_zero_board(self):
        zero_board = np.zeros(self.board_size * self.board_size)
        return zero_board


    # num_reads = _n_playout??
    def UCT_search(self, state_b, num_reads):
        beforeTime = time()
        self._root = TreeNode(None, DummyNode(),self.board_size)
        for _ in range(num_reads):
            state = copy.deepcopy(state_b)
            root = self._root
            leaf = root.select_leaf(state)  # leaf : 노드 객체
            # child_priors가 결국 (82,)가 되든 (81,)가 되든 해야됨
            child_priors, value_estimate = self._policy(state)  # NeuralNet.evaluate(leaf.game_state)
            end, winner = state.game_end()
            if end:  # 누군가 이기거나 draw
                # for end state，return the "true" leaf_value
                # winner은 무승부의 경우 -1이고, 우승자가 존재하면 우승자 int (0,1이였나 1,2였나)
                if winner == -1:  # tie (무승부)
                    value_estimate = 0.0  # 무승부의 경우 leaf_value를 0으로 조정 (value_estimate = leaf_value)
                else:
                    value_estimate = (1.0 if winner == state.get_current_player() else -1.0)  # 우승자가 자신이라면, leaf_value는 1로, 패배자라면 -1로
                leaf.backup(value_estimate)
                continue  # continue한다는건 한판 더 한다는 것
            else:  # 게임에서 못이긴 경우
                leaf.expand(child_priors)
                leaf.backup(value_estimate)
        if self.is_test_mode:
            currentTime = time()
            print(f'UCT-searchTime : {currentTime-beforeTime}')
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
        for i in range(self.board_size*self.board_size):
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
            self._root = TreeNode(None, DummyNode(),self.board_size)

    def __str__(self):
        return "MCTS"


class MCTSPlayerNew(object):
    def __init__(self, policy_value_function, board_size,
                 c_puct=5, n_playout=2000, is_selfplay=0, is_test_mode=False):
        # 여기서 policy_value_function을 가져오기 때문에 어떤 라이브러리를 선택하냐에 따라 MCTS속도가 달라짐
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, is_test_mode=is_test_mode, board_size=board_size)
        self._is_selfplay = is_selfplay
        self.is_test_mode = is_test_mode

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        # np.zeros : 0으로만 채워진 배열 생성하는 함수
        move_probs = np.zeros(board.width * board.height)
        if board.width * board.height - len(board.states) > 0:  # 보드판이 꽉 안찬 경우
            # acts와 probs에 의해 착수 위치가 정해진다.
            time_get_probs = time()  # probs를 얻는데까지 걸리는 시간
            acts, probs = self.mcts.get_move_probs(board, temp)
            if self.is_test_mode:
                time_gap = time() - time_get_probs
                print(f'get_probs 하는데 소요된 시간 : {time_gap}')
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # (자가 학습을 할 때는) Dirichlet 노이즈를 추가하여 탐색
                # 학습할 때 랜덤성이 추가 되는 부분 link221007
                move = np.random.choice(acts, p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                time_update_with_move = time()
                self.mcts.update_with_move(move)
                if self.is_test_mode:
                    print(f'update_with_move 하는데 소요된 시간 : {time() - time_update_with_move}')
            else:  # 플레이어와 대결하는 경우
                # np.random.choice(튜플, int size, boolean replace, array probs) :
                # 아래에서는 size 파라미터를 전달 안했기 때문에 한개만 고른다
                # 플레이어 대결 모드에서는, probs 배열속 남은 자리에서 오직 선택된 하나만 1의 확률을 가지고 나머지는 0을 가지기 떄문에 무조건 정해진 한 위치만 뽑힌다
                # 따라서 아래에 random 키워드가 있다고 해서 이게 임의로 뽑는건 아니다
                # 위에 probs를 할당 받을 때 이미 어디를 고를지는 이미 정해져있다
                # 자가 대결의 경우, 플레이어 대결과는 다르게 np.random.choice()를 수행할 때 dirichlet 노이즈를 통해서 랜덤성을 부여한다
                move = np.random.choice(acts, p=probs)  # link2210172129
                print("ai가 고른 자리 : ",move)
                # 점검
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board_img is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
