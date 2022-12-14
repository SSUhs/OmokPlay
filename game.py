# -*- coding: utf-8 -*-
import numpy as np
import csv
from datetime import datetime
from time import time
from constant import error_const
from rule.renju_rule.renju_rule import Renju_Rule
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
        self.is_train_set_mode = kwargs.get('is_train_set_mode')  # 훈련 데이터 셋을 이용해 학습된 모델로 플레이 하는 경우

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

    # numpy 형태로 판을 받는다
    def get_states_by_numpy(self):
        return np.array(self.states_loc)

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


    # do_move가 수행되기 전에 금수는 되어있어야함!!
    # stone : 흑이 1 백이 2
    def do_move(self, move,stone):
        self.states[move] = stone
        loc = self.move_to_location(move)
        self.states_loc[loc[0]][loc[1]] = stone
        # self.states_loc[loc[0]][loc[1]] = 1 if self.is_you_black() else 2
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move, self.last_loc = move, loc


    def do_undo(self,last_move,before_move):
        if last_move != self.last_move:
            print("오류 : last_move가 일치하지 않습니다")
            quit()
        self.states[last_move] = 0 # 빈 자리로 변경
        loc = self.move_to_location(last_move)
        self.states_loc[loc[0]][loc[1]] = 0
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        if before_move == None: # 이전 움직임이 없을 경우
            self.last_move = -1
            self.last_loc = -1
        else:
            self.last_move = before_move
            self.last_loc = self.move_to_location(before_move)





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
            stone_color = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, stone_color

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, stone_color

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, stone_color

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, stone_color

        return False, -1

    # return : 게임이 끝났으면 (승,패,무승부) True 리턴 / 이긴 돌의 색 리턴 (흑 1 / 백 2 / 무승부 또는 게임이 안끝난 경우 -1)
    def game_end(self):
        win, winner_stone = self.has_a_winner()
        if win:
            return True, winner_stone
        # len(self.states)는 예를 들어 흑 백 다 합쳐서 3번 놨다면 len(self.states)는 3이다
        # 따라서, 오목판 꽉 찼으면 (True,-1) 리턴
        elif len(self.states) == self.width * self.height:
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    # stone_to_forbid : 금수를 정할 돌
    # 주의 : 이 함수는 멤버를 수정하기 때문에 금수 위치가 변경될 수 있음
    #       현재 인스턴스의 states를 기준으로 하기 때문에, 노드 확장으로 다음 예측에는 사용 X
    def set_forbidden_new(self,stone_to_forbid):
        rule = Renju_Rule(self.states_loc, self.width)
        if stone_to_forbid == 1: # 렌주룰에서는 흑만 막는다
            self.forbidden_locations, self.forbidden_types = rule.get_forbidden_points(stone=1)
        elif stone_to_forbid == 2: # 렌주룰에서 백은 금수 X (장목 가능)
            self.forbidden_locations = []
            self.forbidden_types = []
        else:
            print("잘못된 stone number")
            quit()
        self.forbidden_moves = [self.location_to_move(loc) for loc in self.forbidden_locations]

    def set_forbidden(self): # 주의!! 얘는 GUI 설정은 안해주므로, 이거 수행 후에 그래픽에는 따로 금지 이미지 넣어야함
        # forbidden_locations : 흑돌 기준에서 금수의 위치
        rule = Renju_Rule(self.states_loc, self.width)
        self.forbidden_locations, self.forbidden_types = rule.get_forbidden_points(stone=1)
        # if self.order == 0: # order가 0이면 플레이어가 선공
        #     self.forbidden_locations, self.forbidden_types = rule.get_forbidden_points(stone=1)
        # else:
        #     self.forbidden_locations, self.forbidden_types = rule.get_forbidden_points(stone=2)
        self.forbidden_moves = [self.location_to_move(loc) for loc in self.forbidden_locations]


    # 이건 이 함수를 호출한 상황에서 black인지 리턴하는 것. 사람이 흑인지 사람이 백인지를 판단하는 것이 아님
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
    def __init__(self, board, is_gui_mode,game_mode,recorded_game=None,**kwargs):
        self.board = board
        self.players = None
        self.player1 = None
        self.player2 = None
        self.is_gui_mode = is_gui_mode
        self.is_console_mode = not is_gui_mode
        self.gui_board = None
        self.is_initiated_play = False  # init_play()함수 수행 했는지
        self.is_human_intervene = False
        self.game_mode = game_mode
        self.record_list = []
        self.recorded_game = recorded_game


    def graphic_console(self, board, player1, player2,stone,black_white_ai): # 콘솔에 출력
        if not self.is_initiated_play:
            print("init_play()함수를 먼저 호출해야합니다")
            return

        width = board.width
        height = board.height

        clear_output(wait=True)
        os.system('cls')
        print()

        if black_white_ai == 'black':
            print("흑돌(●) : AI")
            print("백돌(○) : 플레이어")
        else:
            print("흑돌(●) : 플레이어")
            print("백돌(○) : AI")

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
                if p == 1: # 흑돌
                    print('●',end='')
                elif p == 2:
                    print('○',end='')
                elif board.is_you_black() and (i, j) in board.forbidden_locations:
                    print('Ⅹ',end='')
                else:
                    print('　', end='')
                # if p == player1:
                #     print('●' if board.order == 0 else '○', end='')
                # elif p == player2:
                #     print('○' if board.order == 0 else '●', end='')
                # elif board.is_you_black() and (i, j) in board.forbidden_locations:
                #     print('Ⅹ', end='')
                # else:
                #     print('　', end='')
            print()
        if board.last_loc != -1:
            print(f"마지막 돌의 위치 : ({board.last_loc[0]},{board.last_loc[1]})\n")

    def graphic_gui(self, gui_board, player1, player2):  # GUI로 출력
        # gui_board.update_game_view(player1, player2)
        gui_board.update_game_view()

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
        # print("init play() 수행")


    def get_move(self,player_in_turn,is_gui,is_computer,row=-1,col=-1,black_white_ai=None):
        move = None
        is_console = not is_gui
        if is_console:
            if is_computer:
                move = player_in_turn.get_action(self.board, black_white_ai)  # AI일 떄는 player_in_turn 인스턴스의 소속 클래스가 MCTSPlayer가 된다
            else: # 콘솔 + 사람
                move = player_in_turn.get_action_console(self.board)
        elif is_gui and is_computer and (not self.board.is_train_set_mode):
            move = player_in_turn.get_action(self.board,is_human_intervene=self.is_human_intervene,black_white_ai=black_white_ai)
        # GUI + 훈련 셋
        elif is_gui and is_computer and self.board.is_train_set_mode:
            if black_white_ai is None:
                print(black_white_ai, "를 설정해주세요")
                quit()
            move = player_in_turn.get_action(self.board,black_white_ai)
        elif is_gui and (not is_computer):
            move = player_in_turn.get_action_gui(self.board, row, col)
        else:
            print("현재 지원되지 않는 경우")
            quit()
        return move

    # console 모드의 경우 row, col을 -1을 대입하면 됨
    # 또한 do_next를 실행하는 차례가 컴퓨터일 경우에도 row,col 사용 X
    # stone : 이번 차례 돌
    def do_next(self, row=None, col=None,stone=None,black_white_ai=None):
        gui_board = self.gui_board
        # 흑돌일 때, 금수 위치를 넣어두기
        # gui로 플레이할 때와 콘솔로 플레이할 때는 do_next를 호출하는 선후가 다르기 때문에 따로 설정
        if stone is None:
            print("stone은 None일 수 없습니다")
            quit()
        if self.is_console_mode:
            if stone == 1:
                self.board.set_forbidden()
            self.graphic_console(self.board, self.player1.player, self.player2.player,stone,black_white_ai)  # 콘솔 모드의 경우, 입력 하기 전에 먼저 콘솔 출력

        current_player = self.board.get_current_player() # AI vs Player 모드 : 1은 사람, 2는 컴퓨터
        player_in_turn = self.players[current_player]
        move = None

        if self.is_console_mode: # 콘솔 모드
            is_computer = False if current_player == 1 else True
            move = self.get_move(player_in_turn, is_gui=False, is_computer=is_computer,black_white_ai=black_white_ai)
        else: # GUI 모드
            if self.game_mode == 'ai_vs_player':
                if current_player == 2:  # 컴퓨터
                    if black_white_ai == 'black' and stone==1:  # AI가 흑이면 AI가 자리를 찾기 전에 먼저 금수 자리 설정
                        self.board.set_forbidden_new(1)
                    move = self.get_move(player_in_turn, is_gui=True, is_computer=True, black_white_ai=black_white_ai)
                else:  # 사람
                    move = self.get_move(player_in_turn, is_gui=True, is_computer=False, row=row, col=col)
                    if move == error_const.CONST_WRONG_POSITION or move == error_const.CONST_BANNED_POSITION or move == error_const.CONST_UNKNOWN:
                        return move  # 잘못된 경우이므로 종료
            elif self.game_mode == 'player_vs_player':
                move = self.get_move(player_in_turn, is_gui=True, is_computer=False, row=row, col=col)
                if move == error_const.CONST_WRONG_POSITION or move == error_const.CONST_BANNED_POSITION or move == error_const.CONST_UNKNOWN:
                    return move  # 잘못된 경우이므로 종료

        self.board.do_move(move,stone)

        if self.is_gui_mode: 
            # 한턴 끝나면 흑의 금수 설정 (이미 돌을 AI or 사람이 돌을 놓은 상태에서 진행되는 부분)
            self.record_move(move) 
            self.board.set_forbidden_new(1)
            self.graphic_gui(gui_board,self.player1.player, self.player2.player)

        end, winner_stone = self.board.game_end()
        if end:
            if self.is_gui_mode:
                self.graphic_gui(gui_board, self.player1.player, self.player2.player)
                if winner_stone != -1:
                    if winner_stone == 1:
                        Tk().wm_withdraw()  # to hide the main window
                        messagebox.showinfo('게임 종료', '흑이 승리하였습니다')
                    else:
                        Tk().wm_withdraw()  # to hide the main window
                        messagebox.showinfo('게임 종료', '백이 승리하였습니다')
                else:  # end 값이 -1인 경우, 무승부 ( game_end() 함수에서, 오목 판에 수들이 꽉차면 -1 리턴해줌)
                    Tk().wm_withdraw()  # to hide the main window
                    messagebox.showinfo('게임 종료', '무승부 입니다')
                self.save_csv_data(winner_stone)
            else:
                self.graphic_console(self.board, self.player1.player, self.player2.player,stone,black_white_ai)
                if winner_stone -1:
                    print("무승부입니다")
                elif winner_stone == 1:
                    print("흑이 승리하였습니다")
                else:
                    print("백이 승리하였습니다")
            return error_const.CONST_GAME_FINISH
        else:
            return error_const.CONST_SUCCESS  # 다시 반복



    # csv로 기록
    def record_move(self, move):
        if move < 0: # 오류는 기록 X
            return
        self.record_list.append(move)
        return

    # 기록한 기보를 CSV 형태로 저장
    def save_csv_data(self,winner_stone):
        if self.game_mode == 'replay':
            return
        current_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        output_csv_name = f'saved_csv/{current_time}.csv'
        winner = None
        if winner_stone == -1:
            winner = 'Draw'
        elif winner_stone == 1:
            winner = 'Black'
        elif winner_stone == 2:
            winner = 'White'
        else:
            print(f"잘못된 인자 : {winner_stone}")
            quit()
        # f_csv = open(output_csv_name, 'w', encoding='utf-8', newline='')
        f_csv = open(output_csv_name, 'w', encoding='utf-8',newline='')
        f_csv_writer = csv.writer(f_csv)
        f_csv_writer.writerow(['Game Mode : ',self.game_mode])
        f_csv_writer.writerow(['Board Size : ',self.board.width])
        f_csv_writer.writerow(['Winner : ',winner])
        f_csv_writer.writerow(self.record_list)
        print(f"CSV 저장 : {output_csv_name}")

    # 이 함수는 "1판" 자가 대전 시간이다
    # 따라서, playout이 400이면 이 함수는 400번 호출됨
    def start_self_play(self, player, is_shown=0, temp=1e-3,is_test_mode=False):
        print("자가 학습 수정중")
        quit()
        """ 스스로 자가 대국하여 학습 데이터(state, mcts_probs, z) 생성 """
        """ 이 함수는 "1판" 자가 대전 """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        before_time = time()
        while True:
            # 흑돌일 때, 금수 위치 확인하기
            if self.board.is_you_black():
                stone = 1
                self.board.set_forbidden()
            else:
                stone =2
            if is_shown:
                self.graphic_console(self.board, p1, p2)

            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)

            # perform a move
            self.board.do_move(move,stone)

            end, winner_stone = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner_stone != -1:
                    print("리턴 값이 바뀌어서 이 부분 수정 필요")
                    quit()
                    # winners_z[np.array(current_players) == winner] = 1.0
                    # winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    self.graphic_console(self.board, p1, p2)
                    if winner_stone != -1:
                        win_color = None
                        if winner_stone == 1:
                            win_color = 'Black'
                        elif winner_stone == 2:
                            win_color = 'White'
                        print("Game end. Winner is player : ", win_color)
                    else:
                        print("Game end. Tie")

                if is_test_mode: print(f"자가대전 \'1\'판 하는데 소요된 시간 : {time()-before_time}")
                return None
                # return winner, zip(states, mcts_probs, winners_z)
