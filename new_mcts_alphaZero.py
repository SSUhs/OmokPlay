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
        print("expand 전 shpae : ",self.child_priors.shape) # (82,)
        self.child_priors = child_priors
        print("expand 후 shpae : ",self.child_priors.shape)  # (1, 81)

        # for action, prob in child_priors:  # enumerate 없이?
        #     # 흑돌일 때 금수 위치는 확장노드에 집어 넣지 않음
        #     if is_you_black and action in forbidden_moves: continue
        #     if action not in self._children:
        #         self.add_child(action, prior=prob)
        #         # self._children[action] = TreeNode(self, prob)  # TreeNode() 에서 self는 parent node를 의미한다


    # def select(self, c_puct):  # select_leaf??
    #     # 자식 노드 중에서 가장 적절한 노드를 선택 한다 (action값)
    #     """Select action among children that gives maximum action value Q plus bonus u(P).
    #     Return: A tuple of (action, next_node)
    #     """
    #     return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    # def update(self, leaf_value):  # backUP?
    #     """Update node values from leaf evaluation.
    #     leaf_value: the value of subtree evaluation from the current player's perspective.
    #     리프 노드까지 다 진행 후에 무승부가 나는 경우 leaf_value가 0이 되고, 패배하면 -1이 되고, 이기면 1이 된다
    #     https://glbvis.blogspot.com/2021/05/ai_20.html
    #     여기 중간 그림 보면 Terminal State에서 1/-1 나와 있다
    #     (update_recursive()를 수행할 때 leaf_value에다가 양음 바꿔서 처리)
    #     """
    #     # 방문 횟수 체크 (평균 계산을 위해서 방문 노드 수 체크)
    #     self.number_visits += 1
    #     # Update Q, a running average of values for all visits.
    #     self._Q += 1.0 * (leaf_value - self._Q) / self.number_visits

    # 자식 노드부터 부모 노드까지 가치값 업데이트
    # def update_recursive(self, leaf_value):
    #     """Like a call to update(), but applied recursively for all ancestors."""
    #     # If it is not root, this node's parent should be updated first.
    #     # if 뒤에 객체가 오는 경우 : __bool__이 오버라이딩 되어 있지 않다면, None이면 false리턴
    #     # 따라서 아래의 조건문을 만족 시키는 경우, 부모 노드가 존재하는 것이므로 부모 노드부터 업데이트 수행
    #     # 아래 조건문이 false라면 부모 노드가 없는 노드이므로 root 노드
    #     if self._parent:
    #         self._parent.update_recursive(-leaf_value)
    #     self.update(leaf_value)

    def select_leaf(self, state):
        current = self
        # print(f'select_leaf type : {type(current)}')
        while current.is_expanded:
            current.number_visits += 1  # Optimizing for performance using NumPy
            current.total_value -= 1  # Optimizing for performance using NumPy
            best_move = current.best_child()
            current = current.maybe_add_child(best_move,state.forbidden_moves, state.is_you_black())
            state.do_move(best_move)  # 이거 while문 밖으로 나가야하나..?
        return current

    # def add_child(self, move, prior):  # move = action (아마 int)
    #     self._children[move] = TreeNode(move, parent=self)

    # maybe가 붙는 이유가, move(action)이 self._children 안에 없는 경우에만 적용되기 떄문인듯
    def maybe_add_child(self, move,forbidden_moves,is_you_black):
        if move not in self._children:
            print("move는 action이므로 type이 int가 나와야함. 그리고 0~225사이값")
            print(f'move type : {type(move)} / move값 : {move}')
            if not (is_you_black and move in forbidden_moves):
                self._children[move] = TreeNode(move,parent=self,board_size=self.board_size)
                # 흑돌일 때 금수 위치는 확장노드에 집어 넣지 않음
        return self._children[move]

    # def get_value(self, c_puct):
    #     """Calculate and return the value for this node.
    #     It is a combination of leaf evaluations Q, and this node's prior adjusted for its visit count, u.
    #     c_puct: a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
    #     """
    #     self._u = (c_puct * self._P *
    #                np.sqrt(self._parent.number_visits) / (1 + self.number_visits))
    #     return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        # leaf node : 자식 노드가 없는 노드
        return self._children == {}

    def is_root(self):
        return self._parent is None

    @property
    def number_visits(self):
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
        print(f'type : 더미노드 child_total_value: {type(self.child_total_value)}')
        print(f'type : 더미노드 child_number_visits: {type(self.child_number_visits)}')

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

    # state : 현재 상태에서 deepcopy 된 state
    # 이 함수는 사용자와의 대결에도 사용 된다
    # 각 상태에서 끝까지 플레이 해본다
    # 이 함수가 n_playout 만큼 돌아가는 것 (디폴트 : 400번)
    # def _playout(self, state):
    #     """Run a single playout from the root to the leaf, getting a value at
    #     the leaf and propagating it back through its parents.
    #     State is modified in-place, so a copy must be provided.
    #     """
    #     node = self._root
    #     while (1):
    #         # 리프 노드가 나올 때까지 계속 진행
    #         # 확장은 여기서 안하고 아래 쪽에 node_expand 에서 진행한다
    #         if node.is_leaf(): break
    #         # Greedily select next move.
    #         action, node = node.select(self._c_puct)
    #         # 현재 state 객체는 _playout 함수 실행하기 전에 deepcopy를 해놓은 state
    #         # 따라서 전달받은 state 상황에서 do_move를 리프노드가 나올 때 까지 쭉 수행해보는 것
    #         # 다 이동 하면 현재 while 문이 종료되고, policy에 의해 판별
    #         state.do_move(action)
    #
    #     # Evaluate the leaf using a network which outputs a list of
    #     # (action, probability) tuples p and also a score v in [-1, 1]
    #     # for the current player.
    #
    #     # 아래 _policy 한번 수행하면 신경망 한번 통과하는 것
    #     # 근데 통과 했는데 게임 종료 상황(누구 한명이 이기거나 비긴 상황)이 아니면 expand를 수행한다
    #     action_probs, leaf_value = self._policy(
    #         state)  # child_priors, value_estimate = NeuralNet.evaluate(leaf.game_state)  # 정책에 따라 행동들의 확률 배열 리턴
    #     # end (bool 타입) : 게임이 단순히 끝났는지 안끝났는지 (승,패 또는 화면 꽉찬 경우에도 end = True)
    #     end, winner = state.game_end()
    #     if not end:  #
    #         node.expand(action_probs, state.forbidden_moves, state.is_you_black())
    #         node.backup(leaf_value)  # value_estimate
    #     else:
    #         # for end state，return the "true" leaf_value
    #         # winner은 무승부의 경우 -1이고, 우승자가 존재하면 우승자 int (0,1이였나 1,2였나)
    #         if winner == -1:  # tie (무승부)
    #             leaf_value = 0.0  # 무승부의 경우 leaf_value를 0으로 조정
    #         else:
    #             leaf_value = (
    #                 1.0 if winner == state.get_current_player() else -1.0)  # 우승자가 자신이라면, leaf_value는 1로, 패배자라면 -1로
    #     # 여기서 -1 하는 이유...?
    #     # node.update_recursive(-leaf_value)
    #     # backup
    #     # 여기에 backup??

    # num_reads = _n_playout??
    def UCT_search(self, state, num_reads):
        root = self._root
        beforeTime = time()
        for _ in range(num_reads):
            leaf = root.select_leaf(state)
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
            leaf.expand(child_priors)
            leaf.backup(value_estimate)
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

        self.UCT_search(copy.deepcopy(state), self._n_playout)

        # for n in range(self._n_playout): 여기 제거
        #     state_copy = copy.deepcopy(state)
        #     # state를 완전히 복사해서 play
        #     self._playout(state_copy)

        act_visits = [(act, node.number_visits) for act, node in self._root._children.items()]
        # print([(state.move_to_location(m),v) for m,v in act_visits])

        # acts = 위치번호 / visits = 방문횟수
        acts, visits = zip(*act_visits)
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
                if self.is_test_mode: print(f'update_with_move 하는데 소요된 시간 : {time() - time_update_with_move}')
            else:  # 플레이어와 대결하는 경우
                # np.random.choice(튜플, int size, boolean replace, array probs) :
                # 아래에서는 size 파라미터를 전달 안했기 때문에 한개만 고른다
                # 플레이어 대결 모드에서는, probs 배열속 남은 자리에서 오직 선택된 하나만 1의 확률을 가지고 나머지는 0을 가지기 떄문에 무조건 정해진 한 위치만 뽑힌다
                # 따라서 아래에 random 키워드가 있다고 해서 이게 임의로 뽑는건 아니다
                # 위에 probs를 할당 받을 때 이미 어디를 고를지는 이미 정해져있다
                # 자가 대결의 경우, 플레이어 대결과는 다르게 np.random.choice()를 수행할 때 dirichlet 노이즈를 통해서 랜덤성을 부여한다
                move = np.random.choice(acts, p=probs)  # link2210172129
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move

        else:
            print("WARNING: the board_img is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
