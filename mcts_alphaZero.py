import numpy as np
import copy
from time import time

from rule.renju_rule import renju_helper


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def get_move_by_human_algorithm(board, acts, probs,black_white_ai):
    renju_helper.get_human_intervene_move(probs,board,black_white_ai,board.width)
    print(board, acts, probs)
    print("구현 X")
    quit()
    return None, None, None


#     # 먼저, 놓자마자 바로 이길 수 있는 좌표가 있으면 해당 좌표를 선택하면 된다
#
#     can_win_list = player_AI.get_win_list(board, True)  # 바로 이길 수 있는 위치 확인 (0~224 1차원 좌표)
#     print('------------------------------')
#     print(f"이길 수 있는 좌표 : {can_win_list}")
#     if len(can_win_list) >= 1:
#         return random.choice(can_win_list), 1.0  # 어차피 리스트에 있는거 아무거나 놔도 이기므로 하나 랜덤으로 골라서 리턴
#
#     # 이제 상대가 놓으면 바로 이길 수 있는 자리 탐색
#     # can_lost_list나 can_win_list는 금수는 이미 처리하고 리턴됨
#     can_lose_list = player_AI.get_win_list(board, False)  # 상대 입장에서 이기는 거 테스트 (type : (0~224 1차원 좌표))
#     print(f'질 수 있는 좌표  : {can_lose_list}')
#     if len(can_lose_list) >= 1:
#         arr_tmp = probs_tmp[0][can_lose_list]  # 만약 질 수 있는 위치가 두개 이상이라면, 신경망을 통해 나온 결과중 높은 곳으로 지정
#         best_choice_idx = np.argmax(arr_tmp)
#         best_move = can_lose_list[best_choice_idx]
#         return best_move, value_current
#
#     arr_list = board.states_loc
#
#     can_attack_list_43 = get_next_43(size, arr_list, board, is_my_turn=True)  # 확실한 공격이 가능한 경우
#     can_attack_list_open4 = get_next_open4(size, arr_list, board, is_my_turn=True)  #
#     print(f"공격 가능 43 : {can_attack_list_43}")
#     print(f"공격 가능 4open : {can_attack_list_open4}")
#     can_attack_list_33 = []
#     if board.is_you_white():
#         can_attack_list_33 = get_next_33(size, arr_list, board, is_my_turn=True)
#         print(f"공격 가능 33 : {can_attack_list_33}")
#
#     can_attack_list = can_attack_list_43 + can_attack_list_33 + can_attack_list_open4
#     if len(can_attack_list) >= 1:
#         arr_tmp = probs_tmp[0][can_attack_list]
#         best_choice_idx = np.argmax(arr_tmp)
#         best_move = can_attack_list[best_choice_idx]
#         return best_move, value_current
#
#     can_defend_list_43 = get_next_43(size, arr_list, board, is_my_turn=False)  # 상대가 나에게 확실한 공격이 가능한 경우
#     can_defend_list_4 = get_next_open4(size, arr_list, board, is_my_turn=False)
#     print(f"방어 필요 43 : {can_defend_list_43}")
#     print(f"방어 필요 4open : {can_defend_list_4}")
#     can_defend_list_33 = []
#     if board.is_you_black():
#         can_defend_list_33 = get_next_33(size, arr_list, board, is_my_turn=False)
#         print(f"방어 필요 33 : {can_defend_list_33}")
#     can_defend_list = can_defend_list_43 + can_defend_list_33 + can_defend_list_4
#     if len(can_defend_list) >= 1:
#         arr_tmp = probs_tmp[0][can_defend_list]
#         best_choice_idx = np.argmax(arr_tmp)
#         best_move = can_defend_list[best_choice_idx]
#         return best_move, value_current

class TreeNode(object):
    """ MCTS 트리의 노드.
    Q : its own value
    P : prior probability
    u : visit-count-adjusted prior score
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

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
        for action, prob in action_priors:
            # 흑돌일 때 금수 위치는 확장노드에 집어 넣지 않음
            if is_you_black and action in forbidden_moves:
                continue
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

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


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, is_test_mode=False):
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
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.is_test_mode = is_test_mode

    # state : 현재 상태에서 deepcopy 된 state
    # 이 함수는 사용자와의 대결에도 사용 된다
    # 각 상태에서 끝까지 플레이 해본다
    # 이 함수가 n_playout 만큼 돌아가는 것 (디폴트 : 400번)
    def _playout(self, state, stone):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while 1:
            # 리프 노드가 나올 때까지 계속 진행
            # 확장은 여기서 안하고 아래 쪽에 node_expand 에서 진행한다
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            # 현재 state 객체는 _playout 함수 실행하기 전에 deepcopy를 해놓은 state
            # 따라서 전달받은 state 상황에서 do_move를 리프노드가 나올 때 까지 쭉 수행해보는 것
            # 다 이동 하면 현재 while 문이 종료되고, policy에 의해 판별
            stone_tmp = 1 if state.is_you_black() else 2
            state.do_move(action, stone_tmp)  # 리프노드가 나올 때 까지 move

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.

        action_probs, leaf_value = self._policy(state)  # 정책에 따라 행동들의 확률 배열 리턴
        # end (bool 타입) : 게임이 단순히 끝났는지 안끝났는지 (승,패 또는 화면 꽉찬 경우에도 end = True)
        end, winner_stone = state.game_end()
        if not end:  #
            node.expand(action_probs, state.forbidden_moves, state.is_you_black())
        else:
            # for end state，return the "true" leaf_value
            # winner은 무승부의 경우 -1이고, 우승자가 존재하면 우승자 int (0,1이였나 1,2였나)
            if winner_stone == -1:  # tie (무승부)
                leaf_value = 0.0  # 무승부의 경우 leaf_value를 0으로 조정
            else:
                leaf_value = (1.0 if stone == winner_stone else -1.0)  # 우승자가 자신이라면, leaf_value는 1로, 패배자라면 -1로
        # 여기서 -1 하는 이유...?
        node.update_recursive(-leaf_value)

    # 여기서 state는 game.py의 board 객체
    def get_move_probs(self, state, stone, temp=1e-3):
        # 이 for 문은 AI가 돌리는 for문
        # _n_playout 횟수가 찰 때까지 playout 수행
        # 따라서, _n_playout가 높을 수록 수행 횟수가 많아 지므로, 소요 시간이 늘어나고 성능은 늘어남
        # 학습할 때 자가대전 몇판 하냐랑은 다른 것
        # n_playout이 400이면, 400번 수행해서 나온 가중치들을 기준으로 확률 배열 리턴
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            # state를 완전히 복사해서 play
            self._playout(state_copy, stone)

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


class MCTSPlayer(object):
    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0, is_test_mode=False):
        # 여기서 policy_value_function을 가져오기 때문에 어떤 라이브러리를 선택하냐에 따라 MCTS속도가 달라짐
        self.mcts = MCTS(policy_value_function, c_puct, n_playout, is_test_mode=is_test_mode)
        self._is_selfplay = is_selfplay
        self.is_test_mode = is_test_mode
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, black_white_ai, temp=1e-3, return_prob=0, is_human_intervene=False):
        stone = 1 if black_white_ai == 'black' else 2
        # np.zeros : 0으로만 채워진 배열 생성하는 함수
        move_probs = np.zeros(board.width * board.height)
        if board.width * board.height - len(board.states) > 0:  # 보드판이 꽉 안찬 경우
            # acts와 probs에 의해 착수 위치가 정해진다.
            time_get_probs = time()  # probs를 얻는데까지 걸리는 시간
            acts, probs = self.mcts.get_move_probs(board, stone, temp)
            if is_human_intervene:  # 특수상황 알고리즘 개입 (ex : 열린4, 43, 닫힌4 등등)
                move_al = get_move_by_human_algorithm(board, acts, probs,black_white_ai)
                asdf
                if move_al is not None:
                    return move_al

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
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move

        else:
            print("WARNING: the board_img_15 is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
