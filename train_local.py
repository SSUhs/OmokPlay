import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet  # Theano and Lasagne
from datetime import datetime
import pickle
import sys
sys.setrecursionlimit(10**8)

class TrainPipeline():
    def __init__(self, board_width, board_height, train_environment):
        # 훈련 환경 : train_environment = 1 >> 코랩 / = 2 >> 로컬에서 학습
        self.train_environment = train_environment

        # 게임(오목)에 대한 변수들
        self.board_width, self.board_height = board_width, board_height
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board,is_gui_mode = False)
        
        # 학습에 대한 변수들
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # KL에 기반하여 학습 계수를 적응적으로 조정
        self.temp = 1.0  # the temperature param
        # n_playout : 하나의 상태(state)에서 시뮬레이션 돌리는 횟수
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 512  # mini-batch size : 버퍼 안의 데이터 중 512개를 추출
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50  # 지정 횟수마다 모델을 체크하고 저장. 원래는 100이었음 (예를 들어 500이면 self_play 500번마다 파일 한번씩 저장)
        self.game_batch_num = 3000  # 최대 학습 횟수 (게임 한판이 1. 3000이면 3000판 수행)
        self.train_num = 0 # 현재 학습 횟수
        
        # policy-value net에서 학습 시작
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)

        # 훈련할 떄 사용할 플레이어 생성
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        회전 및 뒤집기로 데이터set 확대
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 반시계 방향으로 회전
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 수평으로 뒤집기
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 데이터를 확대
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data) # deque의 오른쪽(마지막)에 삽입

    # 자가 훈련을 통해 정책 업데이트 하는 부분
    # 플레이어와 대결 할 때는 이 함수가 호출 되지 않는다 >> 따라서 플레이어와 AI가 대결할 때는 정책 업데이트 X
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            
            # D_KL diverges 가 나쁘면 빠른 중지
            if kl > self.kl_targ * 4 : break
                
        # learning rate를 적응적으로 조절
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1 : self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10 : self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))

        print(f"kl:{kl:5f}, lr_multiplier:{self.lr_multiplier:3f}, loss:{loss}, entropy:{entropy}, explained_var_old:{explained_var_old:3f}, explained_var_new:{explained_var_new:3f}")

        return loss, entropy

    def run(self):
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            self.train_num += 1
            print(f"batch i:{self.train_num}, episode_len:{self.episode_len}")

            if len(self.data_buffer) > self.batch_size : loss, entropy = self.policy_update()

            # 현재 model의 성능을 체크, 모델 속성을 저장
            # check_freq 횟수 마다 저장 (check_freq가 50이면 50번 훈련마다 한번씩 저장)
            if (i+1) % self.check_freq == 0:
                print(f"★ {self.train_num}번째 batch에서 모델 저장 : {datetime.now()}")
                # code20221004131321
                # .model 파일은 플레이할 때 사용할 모델 파일이고, pickle 파일은 학습 데이터? (실제 게임에서는 .model, 학습 과정에서는 .pickle을 불러 와야한다)
                if self.train_environment == 1: # 코랩 (구글 드라이브 연동)
                    self.policy_value_net.save_model(f'/content/drive/MyDrive/policy_{self.board_width}_{self.train_num}.model')
                    pickle.dump(self, open(f'/content/drive/MyDrive/train_{self.board_width}_{self.train_num}.pickle', 'wb'), protocol=2)
                    print("구글 드라이브에 학습 파일을 저장하였습니다")
                elif self.train_environment == 2: # 로컬
                    self.policy_value_net.save_model(f'{model_path}/policy_{self.board_width}_{self.train_num}.model')
                    pickle.dump(self, open(f'{train_path}/train_{self.board_width}_{self.train_num}.pickle', 'wb'), protocol=2)
                else:
                    print("존재하지 않는 환경입니다")
                    quit()


if __name__ == '__main__':
    print("학습할 사이즈를 입력해주세요 (ex : 9x9면 9 입력)")
    size = int(input())
    if size < 5 or size > 15:
        print("오목 판의 크기는 5이상 15이하여야 합니다")
        quit()

    print(f"{size}x{size} 환경에서 학습을 진행합니다.")
    train_path = f"./save/train_{size}"
    model_path = f"./save/model_{size}"


    print("실행 환경을 입력해주세요\n1: Colab\n2: Local")
    train_environment = int(input())
    if not train_environment == 1 or train_environment == 2:
        print("존재하지 않는 환경입니다")
        quit()



    print("기존에 학습된 모델을 불러와서 이어서 학습할려면, 해당 횟수를 입력해주세요 (처음 부터 학습할려면 0 입력)")
    print("예시 : policy_9_2500.model 파일을 불러오고 싶다면 \"2500\"을 입력")
    init_num = int(input())

    # 이미 학습된 모델이 없는 경우 새로운 파이프 라인을 생성한다
    if init_num == 0 or init_num == None :
        training_pipeline = TrainPipeline(size,size,train_environment)
    else: # 이미 일부 학습된 모델이 있는 경우 기존 파이프라인을 불러온다
        if train_environment == 1:
            print("아직 코랩에서 pickle 파일로 학습 데이터 불러오는 것 구현 X")
            quit()
            # training_pipeline = pickle.load(open(f'{}'))
        elif train_environment == 2:
            training_pipeline = pickle.load(open(f'{train_path}/train_{size}_{init_num}.pickle', 'rb'))
        else:
            print("존재하지 않는 train_environment 입니다")
            quit()

    print(f"★ 학습시작 : {datetime.now()}")
    training_pipeline.run()
    