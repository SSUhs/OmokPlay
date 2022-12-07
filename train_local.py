
import train_dataset
from datetime import datetime
import pickle
import re  # 정규표현식
import sys

from TrainingPipeline import TrainPipeline

sys.setrecursionlimit(10 ** 8)


if __name__ == '__main__':
    print("\"라이브러리 종류 / 사이즈 / 이어서 할 횟수\" 순서로 입력")
    print("라이브러리 : tensorflow / keras / tfkeras / theano")
    print("예시 : \'tensorflow 13 500\'(=텐서플로우 라이브러리로 13x13 500번 부터)")
    param_list = re.split(r'[ ]+', input())  # 예시) {'tensorflow','13','500'}

    if len(param_list) == 1 and param_list[0] == 'trainset':
        train_dataset.start_train_dataset()
        quit()

    if len(param_list) < 3:
        print("형식이 잘못되었습니다")
        quit()
    elif param_list[0] == '':  # 시작이 공백이면 잘못 넣은 것
        print("첫번째 입력에 공백이 존재합니다. 다시 입력해주세요")
        quit()

    ai_lib = param_list[0]
    if ai_lib == 'tf': ai_lib = 'tensorflow'  # 단축

    size = int(param_list[1])
    if size < 5 or size > 15:
        print("오목 판의 크기는 5이상 15이하여야 합니다")
        quit()

    init_num = int(param_list[2])
    train_environment = 1  # 훈련 환경은 COLAB으로만

    if len(param_list) >= 4:
        if param_list[3] == 'test':
            is_test_mode = True
            is_new_MCTS = False
        else:
            is_test_mode = False
            is_new_MCTS = False
    else:
        is_test_mode = False
        is_new_MCTS = False

    if len(param_list) >= 5:
        if param_list[4] == 'new':  # 새로운 mcts_alphaZero 기반
            is_new_MCTS = True
        else:
            is_new_MCTS = False

    if len(param_list) >= 6:
        if param_list[5] == 'local':
            train_environment = 2  # 혹시 local로 테스트 할 경우


    print(f"{size}x{size} 환경에서 학습을 진행합니다.")
    train_path_theano = f"./save/train_{size}"
    model_path_theano = f"./save/model_{size}"

    if is_new_MCTS:
        if not (ai_lib == 'tensorflow' or ai_lib == 'theano'):
            print("new MCTS는 현재 tensorflow 1.0에서만 사용가능")
            quit()
        print("\n\n!!!! New MCTS 환경으로 학습!!!!\n\n")
        print("\n\n!!!! New MCTS 환경으로 학습!!!!\n\n")
        print("\n\n!!!! New MCTS 환경으로 학습!!!!\n\n")

    if ai_lib == 'theano':
        if train_environment == 1:  # colab + google drive
            if init_num == 0 or init_num == None:
                training_pipeline = TrainPipeline(size, size, train_environment, ai_lib, is_test_mode=is_test_mode,
                                                  is_new_MCTS=is_new_MCTS)
            else:
                training_pipeline = pickle.load(open(f'/content/drive/MyDrive/train_{size}_{init_num}.pickle'), 'rb')
        else:
            if init_num == 0 or init_num == None:
                training_pipeline = TrainPipeline(size, size, train_environment, ai_lib, is_test_mode=is_test_mode,
                                                  is_new_MCTS=is_new_MCTS)
            else:
                training_pipeline = pickle.load(open(f'{train_path_theano}/train_{size}_{init_num}.pickle', 'rb'))
    elif ai_lib == 'tensorflow' or ai_lib == 'tensorflow-1.15gpu':
        if init_num == 0 or init_num == None:
            model_file = None
            tf_lr_data = None
        elif train_environment == 1:  # colab + google drive
            model_file = f'/content/drive/MyDrive/tf_policy_{size}_{init_num}_model'
            tf_lr_data = f'/content/drive/MyDrive/tf_train_{size}_{init_num}.pickle'
        else:  # 로컬
            model_file = f'./model/tf_policy_{size}_{init_num}_model'
            tf_lr_data = f'./model/tf_train_{size}_{init_num}.pickle'
        training_pipeline = TrainPipeline(size, size, train_environment, ai_lib, model_file=model_file,
                                          start_num=init_num, tf_lr_data=tf_lr_data, is_test_mode=is_test_mode,
                                          is_new_MCTS=is_new_MCTS)
    elif ai_lib == 'tfkeras':
        if init_num == 0 or init_num == None:
            model_file = None
            keras_lr_data = None
        elif train_environment == 1:
            model_file = f'/content/drive/MyDrive/tfkeras_policy_{size}_{init_num}_model'
            keras_lr_data = f'/content/drive/MyDrive/tfkeras_train_{size}_{init_num}.pickle'
        else:
            print("학습이 불가능한 환경입니다")
            quit()
        training_pipeline = TrainPipeline(size, size, train_environment, ai_lib, model_file=model_file,
                                          start_num=init_num, keras_lr_data=keras_lr_data, is_test_mode=is_test_mode,
                                          is_new_MCTS=is_new_MCTS)
    elif ai_lib == 'keras':
        if init_num == 0 or init_num == None:
            model_file = None
            keras_lr_data = None
        elif train_environment == 1:
            model_file = f'/content/drive/MyDrive/keras_policy_{size}_{init_num}_model'
            keras_lr_data = f'/content/drive/MyDrive/keras_train_{size}_{init_num}.pickle'
        else:
            print(f"잘못된 라이브러리 : {ai_lib}")
            quit()
        training_pipeline = TrainPipeline(size, size, train_environment, ai_lib, model_file=model_file,
                                          start_num=init_num, keras_lr_data=keras_lr_data, is_test_mode=is_test_mode,
                                          is_new_MCTS=is_new_MCTS)
    else:
        print("없는 경우")
        quit()

    print(f"★ 학습시작 : {datetime.now()}")
    training_pipeline.run()
