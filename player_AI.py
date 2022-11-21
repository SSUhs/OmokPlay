# train set을 사용하는 플레이어
import numpy as np
from time import time
import tensorflow as tf
import keras.backend as K

def convert_to_one_dimension(state):
    return np.concatenate(state)

def reshape_to_15_15_1(data):
    return K.reshape(data,[-1,15,15,1])



class player_AI():
    def __init__(self,size,is_test_mode,black_white,train_num,is_sequential_model=True):
        self.size = size
        self.is_test_mode = is_test_mode
        self.black_white = black_white
        self.model = self.load_model(black_white,train_num)
        self.is_sequential_model = is_sequential_model

    def convert_to_2nd_loc(self,index):  # 2차원 좌표로 변경
        y = index / self.size
        x = index-(index/self.size)
        return x,y

    def load_model(self,black_white,train_num):
        model_file = f'./model_train/tf_policy_{self.size}_{train_num}_{black_white}.h5'
        model = tf.keras.models.load_model(model_file)
        return model

    # n*n 형태를 일차원으로

    def get_action(self,board):
        # state : numpy
        state = board.get_states_by_numpy()
        if self.is_sequential_model:
            print("여기에서 흑은 1, 백은 2로 잘 출력되는지 확인필요!!!!")  # code20221120224154
            input = reshape_to_15_15_1(state)
            probs = self.model.predict(input)
            # code20221120231234
            while True:
                best_index = np.argmax(probs[0])
                # 이미 돌이 있는 자리를 선택하거나 금수에 놓은 경우
                if (best_index in board.states) or (
                        self.black_white == 'black' and (best_index in board.forbidden_moves)):
                    probs[0][best_index] = -1  # 금수 자리는 선택 불가능 하게 설정
                    continue
                else:
                    break
            move = best_index
            x, y = self.convert_to_2nd_loc(move)
            print(f"선택된 move : {move} = ({x},{y}")
            return move
        else:
            asdf





    def set_player_ind(self, p):
        self.player = p