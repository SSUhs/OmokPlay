# # 프로젝트랑 관련 X
#
#

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 사용 X
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # GPU 경고 제거
import tensorflow as tf

input_states = tf.placeholder(dtype=tf.float32, shape=[None, 4, 9, 9])
arr2 = tf(a=input_states, perm=[0, 2, 3, 1])
print()



