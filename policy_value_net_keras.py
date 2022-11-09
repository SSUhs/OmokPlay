from __future__ import print_function
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from keras.utils.layer_utils import get_source_inputs
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
# from keras.layers import Input
# from tensorflow.keras import regularizers

# from keras.engine.topology import Input
# from keras.engine.training import Model
# from keras.layers.convolutional import Conv2D
# from keras.layers.core import Activation, Dense, Flatten
# from keras.layers.merge import Add
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
# import keras.backend as K


import numpy as np
import pickle


def cpu():
    with tf.device('/cpu:0'):
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)


def gpu():
    with tf.device('/device:GPU:0'):
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


class PolicyValueNetKeras():
    """policy-value network """

    def __init__(self, board_width, board_height, compile_env, model_file=None, keras_lr_data=None, keras_init_num=0):
        # check_tensorflow(compile_env)
        print("\n\n---------------------------------------------------")
        print(f'텐서 플로우 버전 : {tf.__version__}\n')
        devices_all = tf.config.list_physical_devices()
        print(f'활성 Device : {devices_all}\n')
        device_name = tf.test.gpu_device_name()
        if not tf.test.is_gpu_available():
            print("GPU를 가속하지 않으면 사용할 수 없습니다")
            quit()
        # GPU를 사용하거나 TPU를 사용하지 않으면 종료 (혹시나 가속기를 안켜놓았을 상황을 방지)
        if device_name != '/device:GPU:0': # GPU 안쓰는 환경인 경우
            print("GPU를 가속하지 않으면 사용할 수 없습니다")
            quit()
            # mode = int(input("현재 GPU 가속이 미사용 상태입니다. CPU 환경으로만 진행하려면 0, 종료할려면 1"))
            # if mode != 0:
            #     print("종료 합니다")
            #     quit()
        else:
            print('Found GPU at: {}'.format(device_name))
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.create_policy_value_net()
        self._loss_train_op()

        self.test_keras_environment()
        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)

    def create_policy_value_net(self):
        """create the policy value network """

        # in_x = network = get_source_inputs((4, self.board_width, self.board_height))

        # conv layers
        in_x = network = tf.keras.Input((4, self.board_width, self.board_height))
        network = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first",
                                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first",
                                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first",
                                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # action policy layers
        policy_net = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first",
                                            activation="relu",
                                            kernel_regularizer=l2(self.l2_const))(network)
        policy_net = tf.keras.layers.Flatten()(policy_net)
        self.policy_net = tf.keras.layers.Dense(self.board_width * self.board_height, activation="softmax",
                                                kernel_regularizer=l2(self.l2_const))(policy_net)
        # state value layers
        value_net = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first",
                                           activation="relu",
                                           kernel_regularizer=l2(self.l2_const))(network)
        value_net = tf.keras.layers.Flatten()(value_net)
        value_net = tf.keras.layers.Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = tf.keras.layers.Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = Model(in_x, [self.policy_net, self.value_net])

        self.model.summary()

        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results

        self.policy_value = policy_value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        # legal_positions = board.availables
        legal_positions = list(set(range(board.width*board.height)) - set(board.states.keys()))
        current_state = board.current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op
        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=opt, loss=losses)

        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def train_step(state_input, mcts_probs, winner, learning_rate):
            state_input_union = np.array(state_input)
            mcts_probs_union = np.array(mcts_probs)
            winner_union = np.array(winner)
            loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input),
                                       verbose=0)
            action_probs, _ = self.model.predict_on_batch(state_input_union)
            entropy = self_entropy(action_probs)
            K.set_value(self.model.optimizer.lr, learning_rate)
            self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
            return loss[0], entropy

        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()
        return net_params




    def test_keras_environment(self):
        import timeit

        device_name = tf.test.gpu_device_name()
        if device_name != '/device:GPU:0':
          print(
              '\n\nThis error most likely means that this notebook is not '
              'configured to use a GPU.  Change this in Notebook Settings via the '
              'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
          raise SystemError('GPU device not found')



        # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
        print(tf.__version__, tf.test.is_gpu_available())
        cpu()
        gpu()
        # Run the op several times.
        print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
              '(batch x height x width x channel). Sum of ten runs.')
        print('CPU (s):')
        cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
        print(cpu_time)
        print('GPU (s):')
        gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
        print(gpu_time)
        print('GPU speedup over CPU: {}x'.format(int(cpu_time / gpu_time)))
        print("위 테스트에서 GPU 결과가 0.1초 아래면 GPU가 미작동 중입니다")
        ok = int(input("정상적으로 작동 중이면 0, 아니면 1을 입력해주세요"))
        if ok == 0:
            quit()

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)


