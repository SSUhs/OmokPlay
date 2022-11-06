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
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 사용 X
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # GPU 경고 제거
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        elif self.compile_env == 'colab': # 코랩
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        elif self.compile_env == 'colab-1.15gpu':  # 코랩 테스트용
            import tensorflow as tf
        else:
            print("잘못된 환경")
            quit()

        self.board_width = board_width
        self.board_height = board_height
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
        self.action_conv_flat = tf.reshape(
                self.action_conv, [-1, 4 * board_height * board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
                                         units=board_height * board_width,
                                         activation=tf.nn.log_softmax)
        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                data_format="channels_last",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
                self.evaluation_conv, [-1, 2 * board_height * board_width])
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
        self.value_loss = tf.losses.mean_squared_error(self.labels,
                                                       self.evaluation_fc2)
        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(
                tf.float32, shape=[None, board_height * board_width])
        self.policy_loss = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.multiply(self.mcts_probs, self.action_fc), 1)))
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
        self.entropy = tf.negative(tf.reduce_mean(
                tf.reduce_sum(tf.exp(self.action_fc) * self.action_fc, 1)))

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
        legal_positions = list(set(range(board.width*board.height)) - set(board.states.keys()))
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
                [self.loss, self.entropy, self.optimizer],
                feed_dict={self.input_states: state_batch,
                           self.mcts_probs: mcts_probs,
                           self.labels: winner_batch,
                           self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path,init_num):
        self.saver.restore(self.session, model_path)



