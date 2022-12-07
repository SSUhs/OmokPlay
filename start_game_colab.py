import player_AI
from Human import Human
from player_AI import MCTSPlayer_TrainSet
from game import Board, Game

# Colab 환경에서 게임 진행

import tensorflow as tf

devices_all = tf.config.list_physical_devices()
print(f'활성 Device : {devices_all}\n')
device_name = tf.test.gpu_device_name()
# GPU를 사용하거나 TPU를 사용하지 않으면 종료 (혹시나 가속기를 안켜놓았을 상황을 방지)
if device_name != '/device:GPU:0':  # GPU 안쓰는 환경인 경우
    print("GPU를 가속하지 않으면 텐서플로우를 사용할 수 없습니다")
    quit()


# board_size = int(input("판 크기를 선택해주세요"))
print("-------------------------------------------")
print("게임은 15x15 크기로 진행됩니다")
hard = None
is_train_set_mode = True
is_human_intervene = None
use_mcts = None
while True:
    print("난이도 : 상 / 중 / 하")
    hard = input()
    if hard == '상':
        is_human_intervene = True
        use_mcts = True
        break
    elif hard == '중':
        is_human_intervene = True
        use_mcts = False
        break
    elif hard == '하':
        is_human_intervene = False
        use_mcts = False
        break
    else:
        print("존재하지 않는 난이도 입니다")


board_size = 15
n_in_row = 5

board_arr = Board(width=board_size, height=board_size, n_in_row=5,
                  is_train_set_mode=is_train_set_mode)
game = Game(board_arr, is_gui_mode=False, is_human_intervene=is_human_intervene,game_mode='ai_vs_player')


order = None
while True:
    print("자신이 선공(흑)인 경우에 0, 후공(백)인 경우에 1을 입력하세요.")
    order = int(input())
    if order not in [0, 1]:
        print("잘못된 입력\n")
        continue
    else:
        break


policy_net = player_AI.load_model_train_set_github(model_type='policy', size=board_size)
value_net = player_AI.load_model_train_set_github(model_type='value', size=board_size)

policy_net.summary()
value_net.summary()
black_white_ai = None
if order == 0:
    black_white_ai = 'white'
elif order == 1:
    black_white_ai = 'black'
computer_player = MCTSPlayer_TrainSet(policy_net=policy_net, value_net=value_net, c_puct=5, n_playout=100, is_selfplay=False,
                                      is_test_mode=False, is_human_intervene=is_human_intervene,
                                      black_white_ai=black_white_ai, use_mcts=use_mcts)
human = Human()

player_AI._play_on_colab = True
player_AI._test_mode = False
game.board.init_board(start_player=order)
game.init_play(gui_board=None,player1=human,player2=computer_player)
stone = 1

while True:
    num = game.do_next(-1, -1,stone=stone,black_white_ai=black_white_ai)
    stone = 2 if stone == 1 else 1
    # if num == -20:
    #     continue
    # else:
    #     print("게임 종료")
