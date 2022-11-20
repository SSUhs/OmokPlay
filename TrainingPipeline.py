import sys
import random
import numpy as np
import re  # 정규표현식
from game import Board, Game
from pandas import DataFrame, Series
from collections import defaultdict, deque
from mcts_alphaZero import MCTSPlayer
from new_mcts_alphaZero import MCTSPlayerNew
from datetime import datetime
import pickle
from time import time
from time import gmtime
from save_data_helper import save_data_helper

sys.setrecursionlimit(10 ** 8)

list_train_num = []
list_loss = []
list_time = []
list_batch_size = []  # 22.11.08 오전 1시 새로 추가


class TrainPipeline():
    def __init__(self, board_width, board_height, train_environment, ai_lib, model_file=None,
                 start_num=0, tf_lr_data=None, keras_lr_data=None, is_test_mode=False,
                 is_new_MCTS=False,is_train_set_mode=False):  # model_file : 텐서플로우 모델 파일
        # 훈련 환경 : train_environment = 1 >> 코랩 / = 2 >> 로컬에서 학습
        self.train_environment = train_environment
        self.tf_lr_data = tf_lr_data
        self.is_test_mode = is_test_mode
        self.model_file = model_file
        self.is_train_set_mode = is_train_set_mode

        # 학습 라이브러리
        self.ai_lib = ai_lib  # tensorflow 또는 theano

        # 게임(오목)에 대한 변수들
        self.board_width, self.board_height = board_width, board_height
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row,is_train_set_mode=is_train_set_mode)
        self.game = Game(self.board, is_gui_mode=False)

        # 학습에 대한 변수들
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # KL에 기반하여 학습 계수를 적응적으로 조정
        self.temp = 1.0  # the temperature param
        # n_playout : 하나의 상태(state)에서 시뮬레이션 돌리는 횟수
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)  # 결과 데이터가 들어가는 버퍼
        self.batch_size = 512  # mini-batch size : 버퍼 안의 데이터 중 512개를 추출 #
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        # 판 크기가 커질 수록 학습 속도가 느려지므로 백업 주기를 낮춤
        if 5 <= board_width < 9:
            self.check_freq = 50
        elif 9 <= board_width < 11:
            self.check_freq = 40
        elif 11 <= board_width < 12:
            self.check_freq = 30
        elif 13 <= board_width < 15:
            self.check_freq = 20
        else:
            self.check_freq = 10
        self.game_batch_num = 3000  # 최대 학습 횟수 (게임 한판이 1. 3000이면 3000판 수행)

        # policy-value net에서 학습 시작
        if ai_lib == 'theano':
            self.train_num = 0  # 현재 학습 횟수
            from policy_value_net_theano import PolicyValueNetTheano  # Theano and Lasagne
            self.policy_value_net = PolicyValueNetTheano(self.board_width, self.board_height)
        elif ai_lib == 'tensorflow' or ai_lib == 'tensorflow-1.15gpu':
            self.train_num = start_num
            from policy_value_net_tensorflow import PolicyValueNetTensorflow
            self.make_tensorflow_checkpoint_auto(start_num)
            if ai_lib == 'tensorflow':
                self.policy_value_net = PolicyValueNetTensorflow(self.board_width, self.board_height,
                                                                 model_file=model_file, compile_env='colab',
                                                                 init_num=start_num)
            elif ai_lib == 'tensorflow-1.15gpu':  # tensorflow-1.15gpu
                self.policy_value_net = PolicyValueNetTensorflow(self.board_width, self.board_height,
                                                                 model_file=model_file, compile_env='colab-1.15gpu',
                                                                 init_num=start_num)
            if not self.tf_lr_data is None:
                try:
                    with open(tf_lr_data, 'rb') as file:
                        saved_data = pickle.load(file)
                        self.learn_rate = saved_data.learn_rate
                        self.lr_multiplier = saved_data.lr_multiplier
                        self.data_buffer = saved_data.data_buffer
                        print("\nTrainPipeLine 데이터를 로딩했습니다")
                        print(f'learning_rate : {self.learn_rate} lr_multiplier : {self.lr_multiplier}')
                except:
                    print("\ntrain_num이 0이 아닌 상황에서 learning_rate 데이터가 존재하지 않거나 로딩에 실패하였습니다")
        elif ai_lib == 'tfkeras':  # tensorflow keras
            self.train_num = start_num
            from policy_value_net_tf_keras import PolicyValueNetTensorflowKeras
            self.policy_value_net = PolicyValueNetTensorflowKeras(self.board_width, self.board_height,
                                                                  compile_env='colab',
                                                                  model_file=model_file, keras_init_num=start_num,
                                                                  keras_lr_data=keras_lr_data)
        elif ai_lib == 'keras':  # keras
            self.train_num = start_num
            from policy_value_net_keras import PolicyValueNetKeras
            self.policy_value_net = PolicyValueNetKeras(self.board_width, self.board_height,
                                                        compile_env='colab',
                                                        model_file=model_file, init_num=start_num)
        else:
            print("존재하지 않는 라이브러리입니다")
            quit()

        # 훈련할 떄 사용할 플레이어 생성
        if is_new_MCTS:  # 테스트용 MCTS
            self.mcts_player = MCTSPlayerNew(self.policy_value_net.policy_value_fn_new, board_size=board_width,
                                             c_puct=self.c_puct,
                                             n_playout=self.n_playout, is_selfplay=1, is_test_mode=is_test_mode)
        else:
            self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                          n_playout=self.n_playout, is_selfplay=1, is_test_mode=is_test_mode)

    def make_tensorflow_checkpoint_auto(self, start_num):  # 구글 드라이브 체크포인트 자동 생성
        if start_num == 0:
            return
        save_path = '/content/drive/MyDrive/checkpoint'
        model_name = f'tf_policy_{self.board_width}_{start_num}_model'
        # str = f'\"model_checkpoint_path: \"/content/drive/MyDrive/{model_name}\"\nall_model_checkpoint_paths: "/content/drive/MyDrive/{model_name}\"\n'
        str2 = f'\"model_checkpoint_path: \"/content/drive/MyDrive/{model_name}\"\n'
        with open(save_path, "w") as f:
            f.write(str2)
            print("체크 포인트 자동 생성")

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
        # 아래 for문을 다 돌면 자가대전을 n_games만큼 돈 것
        # 기본 한판 하게 되어있음
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_test_mode=self.is_test_mode)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)  # 데이터를 뒤집어서 경우의 수를 더 확대
            self.data_buffer.extend(play_data)  # deque의 오른쪽(마지막)에 삽입

    # 자가 훈련을 통해 정책 업데이트 하는 부분
    # 플레이어와 대결 할 때는 이 함수가 호출 되지 않는다 >> 따라서 플레이어와 AI가 대결할 때는 정책 업데이트 X
    def policy_update(self):
        """update the policy-value net"""
        # data_buffer에 들어 있는
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            # loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            loss = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                    self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            # D_KL diverges 가 나쁘면 빠른 중지
            if kl > self.kl_targ * 4: break

        # learning rate를 적응적으로 조절
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
        explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))

        print(
            f"kl:{kl:5f}, lr_multiplier:{self.lr_multiplier:3f}, loss:{loss}, data_buffer size() : {len(self.data_buffer)}, explained_var_old:{explained_var_old:3f}, explained_var_new:{explained_var_new:3f}")

        return loss


    def run_train_set(self,train_set_length,data_x,data_y):
        print(f'\n\n{self.board_width}x{self.board_width} 사이즈는 구글 드라이브 자동 백업이 {self.check_freq}마다 수행됩니다')
        print("[progress] ", end='', flush=True)
        # Loop all batches for training
        avg_cost = 0
        BATCH_SIZE = self.batch_size  # 미니 배치 수

        # for start in range(0,train_set_length,BATCH_SIZE):
        #     winner_batch = np.reshape(winner_batch, (-1, 1)) 여기 수정
        #     loss,  _ = self.session.run(
        #             [self.loss, self.optimizer],
        #             feed_dict={self.input_states: data_x,
        #                        self.mcts_probs: mcts_probs,
        #                        self.labels: winner_batch,
        #                        self.learning_rate: lr})
        #
        # ------ 아래 참고 ------
        # for start in range(0, train_set_length, BATCH_SIZE):
        #     end = min(start + BATCH_SIZE, train_set_length)
        #     batch_x = data_x[start:end]
        #     batch_y = data_y[start:end]
        #     sess.run(train,
        #              feed_dict={X: batch_x, Y: batch_y, dropout_rate: 0.5})
        #     avg_cost += sess.run(cost, feed_dict=data) * len(batch_x) / train_data_len
        #
        # print("", end='\r', flush=True)


    def run(self):
        if self.is_train_set_mode:
            print("train set 훈련은 run_train_set() 으로 실행해주세요")
            quit()
        print(f'\n\n{self.board_width}x{self.board_width} 사이즈는 구글 드라이브 자동 백업이 {self.check_freq}마다 수행됩니다')

        before_time = time()
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            self.train_num += 1
            before_time_gap = time() - before_time
            print(f"\n소요시간 : {before_time_gap}초, 게임 플레이 횟수 :{self.train_num}, episode_len:{self.episode_len}")
            before_time = time()

            if len(self.data_buffer) > self.batch_size:
                loss = self.policy_update()
                add_csv_data(train_num=self.train_num, loss=loss, batch_size=self.batch_size)

            # 현재 model의 성능을 체크, 모델 속성을 저장
            # check_freq 횟수 마다 저장 (check_freq가 50이면 50번 훈련마다 한번씩 저장)
            if (i + 1) % self.check_freq == 0:
                print(f"★ {self.train_num}번째 batch에서 모델 저장 : {datetime.now()}")
                # .model 파일은 플레이할 때 사용할 모델 파일이고, pickle 파일은 학습 데이터? (실제 게임에서는 .model, 학습 과정에서는 .pickle을 불러 와야한다)
                if self.train_environment == 1:  # 코랩 (구글 드라이브 연동)
                    if self.ai_lib == 'theano':
                        self.policy_value_net.save_model(
                            f'/content/drive/MyDrive/policy_{self.board_width}_{self.train_num}.model')
                        pickle.dump(self,
                                    open(f'/content/drive/MyDrive/train_{self.board_width}_{self.train_num}.pickle',
                                         'wb'), protocol=2)  # theano만 pickle로 저장
                        make_csv_file(self.board_width, self.train_num)
                    elif self.ai_lib == 'tensorflow' or self.ai_lib == 'tensorflow-1.15gpu':
                        self.policy_value_net.save_model(
                            f'/content/drive/MyDrive/tf_policy_{self.board_width}_{self.train_num}_model')
                        make_csv_file(self.board_width, self.train_num)
                        data_helper = save_data_helper(self.train_num, self.board_width, self.learn_rate,
                                                       self.lr_multiplier, self.data_buffer)
                        data_helper.save_model_data()  # lr_multiplier 저장
                    elif self.ai_lib == 'keras':
                        self.policy_value_net.save_model(model_file=self.model_file)
                    else:
                        print("사용할 수 없는 라이브러리입니다")
                        quit()
                    print("학습 파일을 저장하였습니다")
                elif self.train_environment == 2:  # 로컬
                    self.policy_value_net.save_model(
                        f'./save/model_{self.board_width}/policy_{self.board_width}_{self.train_num}.model')
                    pickle.dump(self,
                                open(f'./save/model_{self.board_width}/train_{self.board_width}_{self.train_num}.pickle', 'wb'),
                                protocol=2)
                else:
                    print("존재하지 않는 환경입니다")
                    quit()


def add_csv_data(train_num, loss, batch_size):
    list_train_num.append(train_num)
    list_time.append(datetime.now())
    list_loss.append(loss)
    list_batch_size.append(batch_size)


def make_csv_file(board_size, last_train_num):
    df = DataFrame({'train_num': Series(list_train_num), 'time': Series(list_time), 'loss': Series(list_loss),
                    'batch_size': Series(list_batch_size)})
    df.to_csv(f'/content/drive/MyDrive/{board_size}x{board_size}_{last_train_num}.csv', header=False, index=False)
    list_train_num.clear()  # 초기화
    list_time.clear()  # 초기화
    list_loss.clear()  # 초기화
    list_batch_size.clear()