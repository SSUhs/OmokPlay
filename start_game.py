import pickle

import gui_ai_vs_player
import gui_main
import gui_select_hard
from Human import Human
from constant import error_const
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from tkinter import *
from tkinter import messagebox

execution_environment = 1  # 1 : 로컬 + GUI / 2 : 깃허브 원본 / 3 : 로컬 + 콘솔
# ai_library = 'tensorflow' # 사용할 모델이 어떤 라이브러리 학습 되었는지 :  tensorflow 또는 theano

# 콘솔 모드에서 난이도
hard_console = 15000  # [ 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000 ] 학습 수가 많아질 수록 AI의 난이도가 올라간다

model_file_github_original = './omok_AI/model/policy_9_' + str(
    hard_console) + ".model"  # play.ipynb를 통해 https://github.com/gibiee/omok_AI를 clone 해서 실행하는 경우
model_file_local = './model/policy_9_' + str(hard_console) + ".model"


def run():
    num_console = 5
    # s = int(input("오목 판 크기를 입력하세요 : "))
    # if not 5 < s < 16:
    #     print("오목 판 크기는 6부터 15입니다")
    width_console, height_console = 9, 9 # 콘솔 게임 화면 크기
    is_gui_mode = False
    print("\n")
    if execution_environment == 1:
        print("실행 환경 : 로컬 + GUI")
        is_gui_mode = True
    elif execution_environment == 2:
        print("실행환경: 깃허브 원본 + 코랩")
        print("이 오목 인공지능은 %sx%s 환경에서 동작합니다." % (width_console, height_console))
    elif execution_environment == 3:
        print("실행 환경 : 로컬 + 콘솔")
        print("이 오목 인공지능은 %sx%s 환경에서 동작합니다." % (width_console, height_console))
    else:
        print("존재하지 않는 환경입니다")
        return


    model_file = None
    if execution_environment == 1 or execution_environment == 3:
        model_file = model_file_local
    elif execution_environment == 2:
        model_file = model_file_github_original
    else:
        print("없는 모드입니다")
        return

    if is_gui_mode:
        gui = gui_select_hard.Gui()
        gui.run()
        # gui = gui_main.Gui()
        # gui.run()
    else:
        if not ai_library == 'theano':
            print("현재 콘솔 모드 플레이는 theano 라이브러리로 학습된 모델만 사용 가능합니다")
            quit()
        gui_board = None
        board_arr = Board(width=width_console, height=height_console, n_in_row=num_console)
        game = Game(board_arr, is_gui_mode=is_gui_mode)

        print("자신이 선공(흑)인 경우에 0, 후공(백)인 경우에 1을 입력하세요.")
        order = int(input())
        if order not in [0, 1]: return "강제 종료"

        # 이미 제공된 model을 불러와서 학습된 policy_value_net을 얻는다.
        policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
        best_policy = PolicyValueNetNumpy(width_console, height_console, policy_param)

        # n_playout값 : 성능
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
        human = Human()

        # start_player = 0 → 사람 선공 / 1 → AI 선공
        game.board.init_board(start_player=order)
        game.init_play(None, human, mcts_player, is_shown=1)
        while True:
            num = game.do_next(-1, -1)
            if num == -20:
                continue
            else:
                print("게임 종료")





if __name__ == '__main__':
    run()