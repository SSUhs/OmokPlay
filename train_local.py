import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from datetime import datetime
from pandas import DataFrame, Series
import pickle
import sys
sys.setrecursionlimit(10**8)


list_train_num = []
list_loss = []
list_entropy = []
list_time = []
def add_csv_data(train_num,loss, entropy):
    list_train_num.append(train_num)
    list_time.append(datetime.now())
    list_loss.append(loss)
    list_entropy.append(entropy)

def make_csv_file(board_size,last_train_num):
    df = DataFrame({'train_num':Series(list_train_num),'time':Series(list_time),'loss':Series(list_loss),'entropy':Series(list_entropy)})
    df.to_csv(f'/content/drive/MyDrive/{board_size}x{board_size}_{last_train_num}.csv', header=False, index=False)
    list_train_num.clear() # 초기화
    list_time.clear()  # 초기화
    list_loss.clear()  # 초기화
    list_entropy.clear()  # 초기화



class TrainPipeline():
    def __init__(self, board_width, board_height, train_environment,ai_lib,tf_model_file=None,tf_init_num=0): # tf_model_file : 텐서플로우 모델 파일
        # 훈련 환경 : train_environment = 1 >> 코랩 / = 2 >> 로컬에서 학습
        self.train_environment = train_environment

        # 학습 라이브러리
        self.ai_lib = ai_lib # tensorflow 또는 theano

        # 게임(오목)에 대한 변수들
        self.board_width, self.board_height = board_width, board_height
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board,is_gui_mode = False)
        
        # 학습에 대한 변수들 # code20221102130401
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
        # 판 크기가 커질 수록 학습 속도가 느려지므로 백업 주기를 낮춤
        if 5 <= board_width < 9:
            self.check_freq = 50
        elif 9 <= board_width < 11:
            self.check_freq = 5
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
        elif ai_lib == 'tensorflow':
            self.train_num = tf_init_num
            from policy_value_net_tensorflow import PolicyValueNetTensorflow
            self.make_tensorflow_checkpoint_auto(tf_init_num)
            self.policy_value_net = PolicyValueNetTensorflow(self.board_width, self.board_height,model_file=tf_model_file,compile_env='colab',init_num=tf_init_num)
        elif ai_lib == 'tensorflow-1.15gpu':
            self.train_num = tf_init_num
            from policy_value_net_tensorflow import PolicyValueNetTensorflow
            self.make_tensorflow_checkpoint_auto(tf_init_num)
            self.policy_value_net = PolicyValueNetTensorflow(self.board_width, self.board_height,model_file=tf_model_file,compile_env='colab-1.15gpu',init_num=tf_init_num)
        else:
            print("존재하지 않는 라이브러리입니다")
            quit()

        # 훈련할 떄 사용할 플레이어 생성
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)


    def make_tensorflow_checkpoint_auto(self,init_num):  # 구글 드라이브 체크포인트 자동 생성
        if init_num ==0:
            return
        save_path = '/content/drive/MyDrive/checkpoint'
        model_name = f'tf_policy_{self.board_width}_{init_num}_model'
        str = f'\"model_checkpoint_path: \"/content/drive/MyDrive/{model_name}\"\nall_model_checkpoint_paths: "/content/drive/MyDrive/{model_name}\"\n'
        with open(save_path,"w") as f:
            f.write(str)
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


        print(f"kl:{kl:5f}, learn_rate : {self.learn_rate} lr_multiplier:{self.lr_multiplier:3f}, loss:{loss}, entropy:{entropy}, explained_var_old:{explained_var_old:3f}, explained_var_new:{explained_var_new:3f}")


        return loss, entropy

    def run(self):
        print(f'\n\n{self.board_width}x{self.board_width} 사이즈는 구글 드라이브 자동 백업이 {self.check_freq}마다 수행됩니다')
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            self.train_num += 1
            print(f"\n게임 플레이 횟수 :{self.train_num}, episode_len:{self.episode_len}")

            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_update()
                add_csv_data(train_num=self.train_num,loss=loss,entropy=entropy)

            # 현재 model의 성능을 체크, 모델 속성을 저장
            # check_freq 횟수 마다 저장 (check_freq가 50이면 50번 훈련마다 한번씩 저장)
            if (i+1) % self.check_freq == 0:
                print(f"★ {self.train_num}번째 batch에서 모델 저장 : {datetime.now()}")
                # .model 파일은 플레이할 때 사용할 모델 파일이고, pickle 파일은 학습 데이터? (실제 게임에서는 .model, 학습 과정에서는 .pickle을 불러 와야한다)
                if self.train_environment == 1: # 코랩 (구글 드라이브 연동)
                    if self.ai_lib == 'theano':
                        self.policy_value_net.save_model(f'/content/drive/MyDrive/policy_{self.board_width}_{self.train_num}.model')
                        pickle.dump(self, open(f'/content/drive/MyDrive/train_{self.board_width}_{self.train_num}.pickle', 'wb'), protocol=2) # theano만 pickle로 저장
                        make_csv_file(self.board_width,self.train_num)
                    elif self.ai_lib == 'tensorflow' or self.ai_lib == 'tensorflow-1.15gpu':
                        self.policy_value_net.save_model(f'/content/drive/MyDrive/tf_policy_{self.board_width}_{self.train_num}_model')
                        make_csv_file(self.board_width,self.train_num)
                        pickle.dump(self, open(f'/content/drive/MyDrive/tf_train_{self.board_width}_{self.train_num}.pickle', 'wb'), protocol=2)
                    else:
                        print("사용할 수 없는 라이브러리입니다")
                        quit()
                    print("학습 파일을 저장하였습니다")
                elif self.train_environment == 2: # 로컬
                    self.policy_value_net.save_model(f'{model_path_theano}/policy_{self.board_width}_{self.train_num}.model')
                    pickle.dump(self, open(f'{train_path_theano}/train_{self.board_width}_{self.train_num}.pickle', 'wb'), protocol=2)
                else:
                    print("존재하지 않는 환경입니다")
                    quit()





if __name__ == '__main__':
    print("학습할 사이즈를 입력해주세요 (ex : 9x9면 9 입력)\n")
    size = int(input())
    if size < 5 or size > 15:
        print("오목 판의 크기는 5이상 15이하여야 합니다")
        quit()

    print(f"{size}x{size} 환경에서 학습을 진행합니다.")
    train_path_theano = f"./save/train_{size}"
    model_path_theano = f"./save/model_{size}"


    print("실행 환경을 입력해주세요\n1: Colab\n2: Local\n")
    train_environment = int(input())
    if not (train_environment == 1 or train_environment == 2):
        print("존재하지 않는 환경입니다")
        quit()


    print("학습에 이용할 라이브러리를 선택해주세요 : \'tensorflow\' 또는 \'theano\'\n")
    ai_lib = input()
    if ai_lib == 'tf':
        ai_lib = 'tensorflow'

    print("기존에 학습된 모델을 불러와서 이어서 학습할려면, 해당 횟수를 입력해주세요 (처음 부터 학습할려면 0 입력)")
    print("예시 : policy_9_2500.model 파일을 불러오고 싶다면 \"2500\"을 입력  (2500회 학습한 파일)\n")
    init_num = int(input())

    if ai_lib == 'theano':
        if train_environment == 1: # colab + google drive
            if init_num == 0 or init_num == None:
                training_pipeline = TrainPipeline(size, size, train_environment, ai_lib)
            else:
                training_pipeline = pickle.load(open(f'/content/drive/MyDrive/train_{size}_{init_num}.pickle'), 'rb')
        else:
            if init_num == 0 or init_num == None:
                training_pipeline = TrainPipeline(size, size, train_environment, ai_lib)
            else:
                training_pipeline = pickle.load(open(f'{train_path_theano}/train_{size}_{init_num}.pickle', 'rb'))
    elif ai_lib == 'tensorflow' or ai_lib == 'tensorflow-1.15gpu':
        if init_num == 0 or init_num == None:
            tf_model_file = None
        elif train_environment == 1: # colab + google drive
            tf_model_file = f'/content/drive/MyDrive/tf_policy_{size}_{init_num}_model'
        else: # 로컬
            tf_model_file = f'./model/tf_policy_{size}_{init_num}_model'
        training_pipeline = TrainPipeline(size, size, train_environment, ai_lib,tf_model_file=tf_model_file,tf_init_num=init_num)
    else:
        print("없는 경우")
        quit()
    print(f"★ 학습시작 : {datetime.now()}")
    training_pipeline.run()
    #
    # # 이미 학습된 모델이 없는 경우 새로운 파이프 라인을 생성한다
    # if init_num == 0 or init_num == None :
    #     training_pipeline = TrainPipeline(size,size,train_environment,ai_lib)
    # else: # 이미 일부 학습된 모델이 있는 경우 기존 파이프라인을 불러온다
    #     if train_environment == 1: # Colab - 구글 드라이브
    #         if ai_lib == 'tensorflow':
    #
    #         elif ai_lib == 'theano':
    #             training_pipeline = pickle.load(open(f'/content/drive/MyDrive/train_{size}_{init_num}.pickle'), 'rb')
    #         else:
    #             print("없는 경우")
    #             quit()
    #     elif train_environment == 2: # 로컬
    #         if ai_lib == 'tensorflow':
    #             print("로컬 환경에서 학습시, CUDA가 지원되지 않는 PC에서 텐서플로우 라이브러리를 사용하면 학습 속도가 매우 저하될 수 있습니다")
    #             training_pipeline
    #         elif ai_lib == 'theano':
    #             training_pipeline = pickle.load(open(f'{train_path_theano}/train_{size}_{init_num}.pickle', 'rb'))
    #         else:
    #             print("없는 경우")
    #             quit()
    #     else:
    #         print("존재하지 않는 train_environment 입니다")
    #         quit()


    