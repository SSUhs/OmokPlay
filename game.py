# -*- coding: utf-8 -*-
import numpy as np

from constant import error_const
from renju_rule import Renju_Rule
from IPython.display import clear_output
from tkinter import messagebox
from tkinter import *
import os


class Board(object):
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        self.order = start_player  # order = 0 → 사람 선공(흑돌) / 1 → AI 선공(흑돌)
        self.current_player = self.players[start_player]  # current_player = 1 → 사람 / 2 → AI
        self.last_move, self.last_loc = -1, -1

        self.states, self.states_loc = {}, [[0] * self.width for _ in range(self.height)]
        self.forbidden_locations, self.forbidden_moves = [], []
        self.forbidden_types = []

        """
        # 금수 판정 디버그용
        self.states_loc = list(
        [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        
        for i in range(15) :
            for j in range(15) :
                if self.states_loc[i][j] != 0 : self.states[i*15+j] = self.states_loc[i][j]
        """

    def move_to_location(self, move):
        """ 3*3 보드를 예로 들면 : move 5 는 좌표 (1,2)를 의미한다."""  # ex) 0 1 2
        h = move // self.width  # 3 4 5
        w = move % self.width  # 6 7 8
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2: return error_const.CONST_WRONG_POSITION
        h, w = location[0], location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return error_const.CONST_WRONG_POSITION
        return move

    def current_state(self):
        """현재 플레이어의 관점에서 보드 상태(state)를 return한다.
        state shape: 4 * [width*height] """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0  # 내가 둔 돌의 위치를 1로 표현
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0  # 적이 둔 돌의 위치를 1로 표현
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0  # 마지막 돌의 위치

        if len(self.states) % 2 == 0: square_state[3][:, :] = 1.0  # indicate the colour to play

        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        loc = self.move_to_location(move)
        self.states_loc[loc[0]][loc[1]] = 1 if self.is_you_black() else 2
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move, self.last_loc = move, loc

    # 무승부의 경우 -1 리턴하고, 한명이라도 승리한 경우 승리자 번호 (0,1이였나 1,2였나) 리턴
    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        # moved : 이미 돌이 놓인 자리들
        moved = list(self.states.keys())
        if len(moved) < self.n_in_row * 2 - 1: return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    # return : 첫번째 리턴 값은 게임이 끝났냐 안끝났냐고, 두번째 리턴은 승리 했냐 패배했냐 무승부 했냐를 리턴함
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        # len(self.states)는 예를 들어 흑 백 다 합쳐서 3번 놨다면 len(self.states)는 3이다
        # 따라서, 오목판 꽉 찼으면 (True,-1) 리턴
        elif len(self.states) == self.width * self.height:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def set_forbidden(self): # 주의!! 얘는 GUI 설정은 안해주므로, 이거 수행 후에 그래픽에는 따로 금지 이미지 넣어야함
        # forbidden_locations : 흑돌 기준에서 금수의 위치
        rule = Renju_Rule(self.states_loc, self.width)
        if self.order == 0: # order가 0이면 플레이어가 선공
            self.forbidden_locations, self.forbidden_types = rule.get_forbidden_points(stone=1)
        else:
            self.forbidden_locations, self.forbidden_types = rule.get_forbidden_points(stone=2)
        self.forbidden_moves = [self.location_to_move(loc) for loc in self.forbidden_locations]

    def is_you_black(self):
        # order, current_player
        # (0,1) → 사람(흑돌)
        # (0,2) → AI(백돌)
        # (1,1) → 사람(백돌)
        # (1,2) → AI(흑돌)
        if self.order == 0 and self.current_player == 1:
            return True
        elif self.order == 1 and self.current_player == 2:
            return True
        else:
            return False

    def is_you_white(self):
        if self.is_you_black():
            return False
        else:
            return True


class Game(object):
    def __init__(self, board, is_gui_mode, **kwargs):
        self.board = board
        self.players = None
        self.player1 = None
        self.player2 = None
        self.is_gui_mode = is_gui_mode
        self.gui_board = None
        self.is_initiated_play = False  # init_play()함수 수행 했는지

    def graphic_console(self, board, player1, player2):  # 콘솔에 출력
        if not self.is_initiated_play:
            print("init_play()함수를 먼저 호출해야합니다")
            return

        width = board.width
        height = board.height

        clear_output(wait=True)
        os.system('cls')

        print()
        if board.order == 0:
            print("흑돌(●) : 플레이어")
            print("백돌(○) : AI")
        else:
            print("흑돌(●) : AI")
            print("백돌(○) : 플레이어")
        print("--------------------------------\n")

        if board.current_player == 1:
            print("당신의 차례입니다.\n")
        else:
            print("AI가 수를 두는 중...\n")

        row_number = ['⒪', '⑴', '⑵', '⑶', '⑷', '⑸', '⑹', '⑺', '⑻', '⑼', '⑽', '⑾', '⑿', '⒀', '⒁']
        print('　', end='')
        for i in range(height): print(row_number[i], end='')
        print()
        for i in range(height):
            print(row_number[i], end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('●' if board.order == 0 else '○', end='')
                    # if is_gui_mode:
                    #     self.gui_board.do_blit(i,j,"black" if board.order == 0 else "white")
                elif p == player2:
                    print('○' if board.order == 0 else '●', end='')
                elif board.is_you_black() and (i, j) in board.forbidden_locations:
                    print('Ⅹ', end='')
                else:
                    print('　', end='')
            print()
        if board.last_loc != -1:
            print(f"마지막 돌의 위치 : ({board.last_loc[0]},{board.last_loc[1]})\n")

    def graphic_gui(self, gui_board, player1, player2):  # GUI로 출력
        gui_board.update_game_view(player1, player2)

    def init_play(self, gui_board, player1, player2, is_shown=1):
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        self.players = players
        self.player1 = player1
        self.player2 = player2
        self.gui_board = gui_board
        self.is_initiated_play = True
        print("init play() 수행")

    # console 모드의 경우 row, col을 -1을 대입하면 됨
    # 또한 do_next를 실행하는 차례가 컴퓨터일 경우에도 row,col 사용 X
    def do_next(self, row, col):
        gui_board = self.gui_board
        # 흑돌일 때, 금수 위치를 넣어두기
        # gui로 플레이할 때와 콘솔로 플레이할 때는 do_next를 호출하는 선후가 다르기 때문에 따로 설정

        if not self.is_gui_mode:
            if self.board.is_you_black():
                self.board.set_forbidden()
            self.graphic_console(self.board, self.player1.player, self.player2.player)  # 콘솔 모드의 경우, 입력 하기 전에 먼저 콘솔 출력

        current_player = self.board.get_current_player() # 1은 사람, 2는 컴퓨터
        player_in_turn = self.players[current_player]

        move = None


        # if self.board.is_you_black():
        #     self.board.set_forbidden()

        if not self.is_gui_mode: # 콘솔 모드
            if current_player == 1:
                move = player_in_turn.get_action_console(self.board)
            else:
                move = player_in_turn.get_action(self.board) # AI일 떄는 player_in_turn 인스턴스의 소속 클래스가 MCTSPlayer가 된다
        else: # GUI 모드
            if current_player == 2:  # 컴퓨터
                move = player_in_turn.get_action(self.board)
            else:  # 사람
                move = player_in_turn.get_action_gui(self.board, row, col)
                if move == error_const.CONST_WRONG_POSITION or move == error_const.CONST_BANNED_POSITION or move == error_const.CONST_UNKNOWN:
                    return move  # 잘못된 경우이므로 종료

        self.board.do_move(move)

        if self.is_gui_mode:  # gui_mode의 경우 콘솔 출력은 이동 후에 출력
            # if self.board.last_loc != -1:
            #     print(f"마지막 돌의 위치 : ({self.board.last_loc[0]},{self.board.last_loc[1]})\n")
            if self.is_gui_mode:  # 백돌 차례가 끝나고 흑돌차례가 되면 먼저 금지 위치 설정해줘야한다
                if self.board.is_you_black():
                    self.board.set_forbidden()
            self.graphic_gui(gui_board,self.player1.player, self.player2.player)

        end, winner = self.board.game_end()
        if end:
            self.graphic_console(self.board, self.player1.player, self.player2.player)
            self.graphic_gui(gui_board,self.player1.player, self.player2.player)
            if winner != -1:
                print("Game end. Winner is", self.players[winner])
                if self.players[winner] == 1:
                    Tk().wm_withdraw()  # to hide the main window
                    messagebox.showinfo('게임 종료', '컴퓨터가 이겼습니다')
                else:
                    Tk().wm_withdraw()  # to hide the main window
                    messagebox.showinfo('게임 종료', '당신이 이겼습니다')
            else:  # end 값이 -1인 경우, 무승부 ( game_end() 함수에서, 오목 판에 수들이 꽉차면 -1 리턴해줌)
                Tk().wm_withdraw()  # to hide the main window
                messagebox.showinfo('게임 종료', '무승부 입니다')
                print("Game end. Tie")
            return winner
        else:
            return error_const.CONST_SUCCESS  # 다시 반복

    # def start_play_before(self, gui_board, player1, player2, is_shown=1):  # 구버전
    #     # self.board.init_board(start_player)
    #     p1, p2 = self.board.players
    #     player1.set_player_ind(p1)
    #     player2.set_player_ind(p2)
    #     players = {p1: player1, p2: player2}
    #     while True:
    #         # 흑돌일 때, 금수 위치 확인하기
    #         if self.board.is_you_black() : self.board.set_forbidden()
    #         if is_shown:
    #             self.graphic_console(self.board, player1.player, player2.player)
    #             self.graphic_gui(gui_board, self.board, player1.player, player2.player)
    #
    #         current_player = self.board.get_current_player()
    #         player_in_turn = players[current_player]
    #
    #         if current_player == 1 : # 사람일 때
    #             move = player_in_turn.get_action(self.board)
    #         else : # AI일 때
    #             move = player_in_turn.get_action(self.board)
    #
    #         self.board.do_move(move)
    #         end, winner = self.board.game_end()
    #         if end:
    #             if is_shown:
    #                 self.graphic_console(self.board, player1.player, player2.player)
    #                 self.graphic_gui(gui_board,self.board, player1.player, player2.player)
    #                 if winner != -1 : print("Game end. Winner is", players[winner])
    #                 else : print("Game end. Tie")
    #             return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ 스스로 자가 대국하여 학습 데이터(state, mcts_probs, z) 생성 """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            # 흑돌일 때, 금수 위치 확인하기
            if self.board.is_you_black(): self.board.set_forbidden()
            if is_shown:
                self.graphic_console(self.board, p1, p2)


            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # perform a move
            self.board.do_move(move)

            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    self.graphic_console(self.board, p1, p2)
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
