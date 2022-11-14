# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""
# import tensorflow as tf
import numpy as np
# import tensorflow.compat.v1 as tf # 코랩에서 쓰는 경우, 버전 2로 하면 placeholder이 작동 안하므로 변경해줘야함
# import tensorflow as tf
# tf.disable_v2_behavior()
import check_tensorflow


class PolicyValueNetTensorflow():
    def __init__(self, board_width, board_height, model_file=None,compile_env='local', init_num=0):
        self.compile_env=compile_env  # local  / colab
        check_tensorflow.check_tf(compile_env) # 적합한 텐서플로우 버전인지 확인
        if self.compile_env == 'local':  # GPU가 사용 안되는 컴퓨터
            print("환경 : local")
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 사용 X
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # GPU 경고 제거
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        elif self.compile_env == 'colab': # 코랩
            print("환경 : colab")
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        elif self.compile_env == 'colab-1.15gpu':  # 코랩 테스트용
            import tensorflow as tf
        else:
            print("잘못된 환경")
            quit()

        self.board_width = board_width
        self.board_height = board_height
        # tf.keras.layers.Conv2D(32,7)
        # Define the tensorflow neural network
        # 1. Input:
        # 22-10-29 : 텐서플로우 버전이 2.x로 업그레이드 되면서 placeholder 대신에 Variable를 사용해야함 or 버전 2를 비활성화
        # self.input_states = tf.Variable(tf.ones(shape=[None, 4, board_height, board_width]), dtype=tf.float32)
        self.input_states = tf.placeholder(dtype=tf.float32, shape=[None, 4, board_height, board_width])
        self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        # 2. Common Networks Layers
        self.conv1 = tf.layers.conv2d(inputs=self.input_state,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            data_format="channels_last",
                                            activation=tf.nn.relu)

        # Flatten the tensor
        self.action_conv_flat = tf.reshape(self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board;
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(self.evaluation_conv, [-1, 2 * board_height * board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        # 2. Predictions: the array containing the evaluation score of each state
        # which is self.evaluation_fc2
        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels,self.evaluation_fc2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
        # 3-3. L2 penalty (regularization)
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # Define the optimizer we use for training
        self.learning_rate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        # Make a session
        self.session = tf.Session()

        # calc policy entropy, for monitoring only
        # self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver()
        if model_file is not None:
            self.restore_model(model_file,init_num)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
                [self.action_fc, self.evaluation_fc2],
                feed_dict={self.input_states: state_batch}
                )
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        # legal_positions = board.availables
        # legal position : 놓을 수 "있는"포지션
        set_all = set(range(board.width*board.height))
        set_state_keys = set(board.states.keys())
        legal_positions = list(set_all - set_state_keys)
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height)) # 이 current_state는 상대가 놓은 것만 보여주는 듯
        act_probs, value = self.policy_value(current_state)
        # legal_arr : ndarray(225,)부터 시작해서 한 수당 225에서 하나씩 빠짐
        # 참고로, legal_positions은 놓을 수 있는 move(int)임. 얘는 리스트로 하나씩 줄어든다
        # act_probs는 따라서 (일자 좌표, 가중치)로 이루어진 것 >> 3번 좌표로 이동시 가중치 이런 방식임
        # act_probs의 경우, 0~224판까지 모든 경우의 수가 나오는거고,
        # 거기서 legal_positions에 해당하는 것의 개수만큼만 lega_arr로 반환
        # 즉, legal_positions는 한번 수를 놓을 때마다 개수가 줄어드므로 결국 legal_arr 수도 줄어들고,
        # 따라서 최종적으로 리턴하는 tuple의 수도 줄어들게 된다
        legal_arr = act_probs[0][legal_positions] # 얘는 수를 놓을 때마다 사이즈가 줄어 듦  # 왜 0번이냐면 애초에 act_probs가 [1][225] 이런형태라 그럼
        act_probs = zip(legal_positions, legal_arr)
        return act_probs, value


    def policy_value_fn_new(self,board):
        legal_flat_arr = np.zeros(board.width*board.height)
        set_current_keys = set(board.states.keys())
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        for i in range(board.width*board.height):
            if i in set_current_keys:  # 이미 수가 놓아져있는 부분에는 놓을 수가 없으므로 확률을 0으로 조정
                legal_flat_arr[i] = 0
            else:
                legal_flat_arr[i] = 1*(act_probs[0][i])
        return legal_flat_arr, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss,  _ = self.session.run(
                [self.loss, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path,init_num):
        self.saver.restore(self.session, model_path)




