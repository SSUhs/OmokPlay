from constant import error_const
from tkinter import *
from tkinter import messagebox

class Human(object):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    # noinspection PyMethodMayBeStatic
    def get_action_gui(self, board, row, col): # GUI 상황에서 get_action
        location = [row, col]
        try:
            move = board.location_to_move(location)
        except Exception as e:
            move = error_const.CONST_UNKNOWN

        if move == error_const.CONST_WRONG_POSITION or move == error_const.CONST_BANNED_POSITION or move == error_const.CONST_UNKNOWN or move in board.states.keys():  # 이미 바둑알이 놓아져 있는 곳에 놓는 경우
            Tk().wm_withdraw()  # to hide the main window
            messagebox.showinfo('오류', '다시 입력하세요')
            print("오류가 발생한 row, col : ",row,col)
            if move in board.states.keys():
                return error_const.CONST_WRONG_POSITION
            else:
                return move
        elif board.is_you_black() and tuple(location) in board.forbidden_locations:
            # code20221012195307
            Tk().wm_withdraw()  # to hide the main window
            messagebox.showinfo('오류', '금수 자리에 돌을 놓을 수 없습니다')
            return error_const.CONST_BANNED_POSITION
            # move = self.get_action_gui(board)
        return move

    def get_action_console(self, board):  # 콘솔 or Colab용 진행 액션
        try:
            print("돌을 둘 좌표를 입력하세요.")
            location = input()
            if isinstance(location, str): location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = error_const.CONST_UNKNOWN

        if move == -1 or move in board.states.keys():
            print("다시 입력하세요.")
            move = self.get_action_console(board)
        elif board.is_you_black() and tuple(location) in board.forbidden_locations:
            print("금수 자리에 돌을 놓을 수 없습니다.")
            move = self.get_action_console(board)

        return move

    def __str__(self):
        return "Human {}".format(self.player)