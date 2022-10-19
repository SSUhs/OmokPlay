# import pickle
#
# import gui_ai_vs_player
# from constant import error_const
# from game import Board, Game
# from mcts_alphaZero import MCTSPlayer
# from policy_value_net_numpy import PolicyValueNetNumpy
# from tkinter import *
# from tkinter import messagebox
#
# execution_environment = 1  # 1 : 로컬 + GUI / 2 : 깃허브 원본 / 3 : 로컬 + 콘솔
# hard = 15000  # [ 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000 ] 학습 수가 많아질 수록 AI의 난이도가 올라간다
#
# model_file_github_original = './omok_AI/model/policy_9_' + str(
#     hard) + ".model"  # play.ipynb를 통해 https://github.com/gibiee/omok_AI를 clone 해서 실행하는 경우
# model_file_local = './model/policy_9_' + str(hard) + ".model"
#
#
# class Human(object):
#     def __init__(self):
#         self.player = None
#
#     def set_player_ind(self, p):
#         self.player = p
#
#     # noinspection PyMethodMayBeStatic
#     def get_action_gui(self, board, row, col): # GUI 상황에서 get_action
#         location = [row, col]
#         try:
#             move = board.location_to_move(location)
#         except Exception as e:
#             move = error_const.CONST_UNKNOWN
#
#         if move == error_const.CONST_WRONG_POSITION or move == error_const.CONST_BANNED_POSITION or move == error_const.CONST_UNKNOWN or move in board.states.keys():  # 이미 바둑알이 놓아져 있는 곳에 놓는 경우
#             Tk().wm_withdraw()  # to hide the main window
#             messagebox.showinfo('오류', '다시 입력하세요')
#             print("오류가 발생한 row, col : ",row,col)
#             if move in board.states.keys():
#                 return error_const.CONST_WRONG_POSITION
#             else:
#                 return move
#         elif board.is_you_black() and tuple(location) in board.forbidden_locations:
#             # code20221012195307
#             Tk().wm_withdraw()  # to hide the main window
#             messagebox.showinfo('오류', '금수 자리에 돌을 놓을 수 없습니다')
#             return error_const.CONST_BANNED_POSITION
#             # move = self.get_action_gui(board)
#         return move
#
#     def get_action_console(self, board):  # 콘솔 or Colab용 진행 액션
#         try:
#             print("돌을 둘 좌표를 입력하세요.")
#             location = input()
#             if isinstance(location, str): location = [int(n, 10) for n in location.split(",")]
#             move = board.location_to_move(location)
#         except Exception as e:
#             move = error_const.CONST_UNKNOWN
#
#         if move == -1 or move in board.states.keys():
#             print("다시 입력하세요.")
#             move = self.get_action_console(board)
#         elif board.is_you_black() and tuple(location) in board.forbidden_locations:
#             print("금수 자리에 돌을 놓을 수 없습니다.")
#             move = self.get_action_console(board)
#
#         return move
#
#     def __str__(self):
#         return "Human {}".format(self.player)
#
#
# def run():
#     num = 5
#     width, height = 9, 9
#     print("이 오목 인공지능은 %sx%s 환경에서 동작합니다." % (width, height))
#
#     is_gui_mode = False
#     if execution_environment == 1:
#         print("실행 환경 : 로컬 + GUI")
#         is_gui_mode = True
#     elif execution_environment == 2:
#         print("실행환경: 깃허브 원본 + 코랩")
#     elif execution_environment == 3:
#         print("실행 환경 : 로컬 + 콘솔")
#     else:
#         print("존재하지 않는 환경입니다")
#         return
#
#     model_file = None
#     if execution_environment == 1 or execution_environment == 3:
#         model_file = model_file_local
#     elif execution_environment == 2:
#         model_file = model_file_github_original
#     else:
#         print("없는 모드입니다")
#         return
#
#     gui_board = None
#     board_arr = Board(width=width, height=height, n_in_row=num)
#     game = Game(board_arr, is_gui_mode=is_gui_mode)
#
#     print("자신이 선공(흑)인 경우에 0, 후공(백)인 경우에 1을 입력하세요.")
#     order = int(input())
#     if order not in [0, 1]: return "강제 종료"
#
#     # 이미 제공된 model을 불러와서 학습된 policy_value_net을 얻는다.
#     policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
#     best_policy = PolicyValueNetNumpy(width, height, policy_param)
#
#     # n_playout값 : 성능
#     mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
#     human = Human()
#
#     # start_player = 0 → 사람 선공 / 1 → AI 선공
#     game.board.init_board(start_player=order)
#     if is_gui_mode:
#         gui_board = gui_ai_vs_player.Gui(game, board_arr, human, mcts_player)
#         gui_board.run()
#         gui_board.update_game_view()
#     else:
#         game.init_play(None,human, mcts_player, is_shown=1)
#         while True:
#             num = game.do_next(-1, -1)
#             if num == -20:
#                 continue
#             else:
#                 print("게임 종료")
#
#
# if __name__ == '__main__':
#     run()
