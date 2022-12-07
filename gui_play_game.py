import pygame as pg
import ctypes
import rule.renju_rule.renju_helper as renju_helper

from tkinter import messagebox
from tkinter import *

from constant import error_const

ctypes.windll.user32.SetProcessDPIAware()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
size = [800, 800]
button_size = 120
dot_size = 12
const_gap = 51 # 격자 사이 간격

board_img_15 = pg.image.load('images/오목판.png')
board_img_13 = pg.image.load('images/오목판-13.png')
board_img_11 = pg.image.load('images/오목판-11.png')
board_img_9 = pg.image.load('images/오목판-9.png')
stone_black = pg.image.load('images/stone_black2.png')
stone_white = pg.image.load('images/stone_white.png')
mark_33 = pg.image.load('images/33.png')
mark_44 = pg.image.load('images/44.png')
mark_6 = pg.image.load('images/6.png')
img_next = pg.image.load('images/theme_white_gray/다음.png')
img_prev = pg.image.load('images/theme_white_gray/이전.png')
img_background = pg.image.load('images/main_page.png')
# icon = pg.image.load('images/icon5.png')
s = pg.Surface((12, 12))
s.fill((255, 0, 0))
last = pg.Surface((20, 20))

clock = pg.time.Clock()
# pg.display.set_icon(pg.transform.smoothscale(icon, (32, 32)))
pg.display.set_caption("오목")

# 오류 상수들
CONST_WRONG_POSITION = 1


class Gui:
    def __init__(self, game, board_arr, player1, player2, is_test_mode=False, black_white_ai=None, game_mode=None,
                 replay_data=None):
        # self.game_org = game.Game()
        if black_white_ai is None and game_mode == 'ai_vs_player':
            print(black_white_ai, "를 설정해주세요 - gui_ai")
            quit()
        self.game = game
        self.game_mode = game_mode  # ai_vs_player / player_vs_player / replay
        self.player1 = player1
        self.player2 = player2
        self.board_arr = board_arr
        self.diameter_base = None
        self.row_col_mid_const = None
        self.const_size1 = None
        self.width_height_base = None
        board_size = board_arr.width
        if board_size == 15:
            self.width, self.height = 800, 800
            self.width_height_base = 800
            self.diameter_base = 45
            self.row_col_mid_const = 7
        elif board_size == 13:
            self.width, self.height = 671, 673
            self.width_height_base = 672
            self.diameter_base = 45
            # self.diameter_base = 38
            self.row_col_mid_const = 6
        elif board_size == 11:
            self.width, self.height = 576, 575
            self.width_height_base = 576
            self.diameter_base = 45
            # self.diameter_base = 32.4
            self.row_col_mid_const = 5
        elif board_size == 9:
            self.width, self.height = 470, 470
            self.width_height_base = 470
            self.diameter_base = 45
            # self.diameter_base = 26.47
            self.row_col_mid_const = 4

        if game_mode == 'replay':
            self.plus_height = 70
        else:
            self.plus_height = 0
        self.diameter = self.diameter_base
        self.button_size = button_size
        self.dot_size = dot_size
        self.board_img = self.get_board_img()
        self.back_ground = img_background
        self.stone_black = stone_black
        self.stone_white = stone_white
        self.mark_33 = mark_33
        self.img_next = img_next
        self.img_prev = img_prev
        self.mark_44 = mark_44
        self.mark_6 = mark_6
        self.hint = None
        self.is_test_mode = is_test_mode
        self.black_white_ai = black_white_ai
        self.replay_data = replay_data
        self.hint = True if self.is_test_mode else False
        self.replay_turn = 0

        self.new_game = False
        # self.bs = 0
        # self.ws = 0
        self.update_game_view()

    def resize_view(self, event=None):
        if not event is None:
            self.width, self.height = event.dict['size']
        self.diameter = int(self.width / self.height * self.diameter_base)
        self.button_size = int(self.width / self.height * button_size)
        self.dot_size = int(self.width / self.height * dot_size)

        self.back_ground = pg.transform.smoothscale(img_background, (self.width, self.height + self.plus_height))
        if self.game_mode == 'replay':
            self.board_img = pg.transform.smoothscale(self.get_board_img(), (self.width, self.height))
        else:
            self.board_img = pg.transform.smoothscale(board_img_15, (self.width, self.height))
        self.stone_black = pg.transform.smoothscale(stone_black, (self.diameter, self.diameter))
        self.stone_white = pg.transform.smoothscale(stone_white, (self.diameter, self.diameter))
        self.mark_33 = pg.transform.smoothscale(mark_33, (self.diameter, self.diameter))
        self.mark_44 = pg.transform.smoothscale(mark_44, (self.diameter, self.diameter))
        self.mark_6 = pg.transform.smoothscale(mark_6, (self.diameter, self.diameter))
        self.img_next = pg.transform.smoothscale(img_next, (150, 50))
        self.img_prev = pg.transform.smoothscale(img_prev, (150, 50))
        self.update_game_view()

    def run(self):
        done = False
        self.game.init_play(self, self.player1, self.player2, is_shown=1)
        self.resize_view()
        self.update_game_view()
        current_player = self.game.board.get_current_player()  # AI vs Player모드 : 1은 사람, 2는 컴퓨터
        if self.game_mode == 'ai_vs_player' and current_player == 2:  # 사용자가 "백"이고 컴퓨터가 "흑"이면 컴퓨터가 먼저 놓기
            self.game.do_next(stone=1, black_white_ai=self.black_white_ai)  # 컴퓨터 차례이므로 row, col 대입 X
        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                elif event.type == pg.VIDEORESIZE:
                    print("resize event")
                    self.resize_view(event)
                elif event.type == pg.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if self.game_mode == 'replay':
                        self.action_replay(x, y)
                        continue

                    row,col = self.get_row_column(x,y)

                    # row = round((y - 43 * self.width / 800) / (51 * self.width / 800))
                    # col = round((x - 43 * self.width / 800) / (51 * self.width / 800))
                    print("row, col : (", row, ",", col, ")")
                    print(f"클릭 좌표 : (x,y){x},{y}")
                    is_end, winner = self.game.board.game_end()
                    if is_end:
                        Tk().wm_withdraw()
                        messagebox.showinfo('오류', '이미 게임이 종료되었습니다')
                        continue

                    if self.game.board.is_you_black():
                        move = self.game.do_next(row, col, stone=1, black_white_ai=self.black_white_ai)  # 사람이 수행
                    else:
                        move = self.game.do_next(row, col, stone=2, black_white_ai=self.black_white_ai)  # 사람이 수행
                    if move == error_const.CONST_UNKNOWN:
                        Tk().wm_withdraw()  # to hide the main window
                        messagebox.showinfo('오류', '알 수 없는 오류')
                    elif move == error_const.CONST_BANNED_POSITION or move == error_const.CONST_WRONG_POSITION:
                        break
                    elif move == error_const.BANNED_6 or move == error_const.BANNED_33 or move == error_const.BANNED_44:
                        break
                    elif move == error_const.CONST_GAME_FINISH:
                        continue

                    if self.game_mode == 'ai_vs_player':
                        print("AI가 수를 두고 있습니다")
                        if self.black_white_ai == 'black':
                            move_ai = self.game.do_next(row, col, stone=1, black_white_ai=self.black_white_ai)  # AI가 수행
                        else:
                            move_ai = self.game.do_next(row, col, stone=2, black_white_ai=self.black_white_ai)  # AI가 수행
                        print("AI가 수를 두었습니다")

                    elif self.game_mode == 'player_vs_player' and self.is_test_mode:  # 테스트 모드 : 사람한테 힌트
                        next_stone = None
                        next_next_stone = None
                        boards = self.game.board
                        board_size = boards.width
                        arr_list = boards.states_loc
                        next_player = boards.get_current_player()
                        if next_player == 1:
                            next_stone = 1
                            next_stone_str = '흑'
                            next_next_stone_str = '백'
                        else:
                            next_stone = 2
                            next_stone_str = '백'
                            next_next_stone_str = '흑'
                        next_next_stone = renju_helper.get_enemy_stone(next_stone)
                        can_next_win_list = renju_helper.get_win_list(boards, next_stone, board_size)
                        can_next_attack_list_43open = renju_helper.get_next_43open(board_size, arr_list, boards,
                                                                                   next_stone)
                        can_next_attack_list_43closed = renju_helper.get_next_43closed(board_size, arr_list, boards,
                                                                                       next_stone)
                        can_next_attack_list_open4 = renju_helper.get_next_open4(board_size, arr_list, boards,
                                                                                 next_stone)
                        can_next_attack_list_33 = renju_helper.get_next_33(board_size, arr_list, boards, next_stone)
                        can_next_lose_list = renju_helper.get_win_list(boards, next_next_stone, board_size)
                        can_next_defend_list_43open = renju_helper.get_next_43open(board_size, arr_list, boards,
                                                                                   next_next_stone)
                        can_next_defend_list_43closed = renju_helper.get_next_43closed(board_size, arr_list, boards,
                                                                                       next_next_stone)
                        can_next_defend_list_4 = renju_helper.get_next_open4(board_size, arr_list, boards,
                                                                             next_next_stone)
                        can_next_defend_list_33 = renju_helper.get_next_33(board_size, arr_list, boards,
                                                                           next_next_stone)
                        print("")
                        print(f'{next_stone_str} 바로 승리 : {can_next_win_list}')
                        print(f'{next_stone_str} 공격 43open : {can_next_attack_list_43open}')
                        print(f'{next_stone_str} 공격 43closed : {can_next_attack_list_43closed}')
                        print(f'{next_stone_str} 공격 open4 : {can_next_attack_list_open4}')
                        print(f'{next_stone_str} 공격 33 : {can_next_attack_list_33}')
                        print(f'{next_next_stone_str} 바로 승리 : {can_next_lose_list}')
                        print(f'{next_next_stone_str} 공격 43open : {can_next_defend_list_43open}')
                        print(f'{next_next_stone_str} 공격 43closed : {can_next_defend_list_43closed}')
                        print(f'{next_next_stone_str} 공격 open4 : {can_next_defend_list_4}')
                        print(f'{next_next_stone_str} 공격 : {can_next_defend_list_33}')

            # if self.bs != 0 and self.ws != 0:
            #     if self.bs == 2 and self.game.state.check_turn():
            #         #self.game.next(predict_p(self.model, self.game.state))
            #         # self.game.next(mcts_action(self.model, self.model2, self.model3, self.model4, self.game.state))
            #         if self.game.end >= 1:
            #             self.new_game = False
            #             self.bs = 0
            #             self.ws = 0
            #         self.update_game_view()
            #         for event in pg.event.get():
            #             if event.type == pg.QUIT:
            #                 done = True
            #             elif event.type == pg.VIDEORESIZE:
            #                 self.width, self.height = event.dict['size']
            #                 self.diameter = int(self.width / 800 * diameter)
            #                 self.button_size = int(self.width / 800 * button_size)
            #                 self.dot_size = int(self.width / 800 * dot_size)
            #                 self.board_img_15 = pg.transform.smoothscale(board_img_15, (self.width, self.height))
            #                 self.stone_black = pg.transform.smoothscale(stone_black, (self.diameter, self.diameter))
            #                 self.stone_white = pg.transform.smoothscale(stone_white, (self.diameter, self.diameter))
            #                 self.mark_33 = pg.transform.smoothscale(mark_33, (self.diameter, self.diameter))
            #                 self.mark_44 = pg.transform.smoothscale(mark_44, (self.diameter, self.diameter))
            #                 self.mark_6 = pg.transform.smoothscale(mark_6, (self.diameter, self.diameter))
            #                 self.button_1 = pg.transform.smoothscale(button_1, (self.button_size, self.button_size))
            #                 self.button_2 = pg.transform.smoothscale(button_2, (self.button_size, self.button_size))
            #                 self.button_3 = pg.transform.smoothscale(button_3, (self.button_size, self.button_size))
            #                 self.button_4 = pg.transform.smoothscale(button_4, (self.button_size, self.button_size))
            #                 self.button_5 = pg.transform.smoothscale(button_5, (self.button_size, self.button_size))
            #                 self.button_6 = pg.transform.smoothscale(button_6, (self.button_size, self.button_size))
            #                 self.button_7 = pg.transform.smoothscale(button_7, (self.button_size, self.button_size))
            #                 self.update_game_view()
            #             elif event.type == pg.MOUSEBUTTONDOWN:
            #                 print("마우스 좌클릭 - B지점")
            #                 x, y = event.pos
            #                 row = round((y - 43 * self.width / 800) / (51 * self.width / 800))
            #                 col = round((x - 43 * self.width / 800) / (51 * self.width / 800))
            #                 self.game.do_next(row,col)
            #
            #                 # 힌트 버튼
            #                 if 700 * self.width / 800 < x < 780 * self.width / 800 and y < 45 * self.width / 800:
            #                     if self.hint:
            #                         self.hint = False
            #                     else:
            #                         self.hint = True
            #                 # 새 게임 버튼 ? 다시 시작?
            #                 elif 50 * self.width / 800 < x < 130 * self.width / 800 and y < 45 * self.width / 800:
            #                     if not self.new_game:
            #                         self.game.__init__()
            #                         self.bs = 0
            #                         self.ws = 0
            #                         self.new_game = True
            #                     else:
            #                         if self.bs == 0:
            #                             self.bs = 1
            #                         else:
            #                             self.ws = 1
            #                             self.new_game = False
            #                 elif 140 * self.width / 800 < x < 220 * self.width / 800 and y < 45 * self.width / 800:
            #                     if self.new_game:
            #                         if self.bs == 0:
            #                             self.bs = 2
            #                         else:
            #                             self.ws = 2
            #                             self.new_game = False
            #
            #                 self.update_game_view()
            #
            #     elif self.ws == 2 and not self.game.state.check_turn():
            #         #self.game.next(predict_p(self.model, self.model2, self.game.state))
            #         # self.game.next(mcts_action(self.model, self.model2, self.model3, self.model4, self.game.state))
            #         if self.game.end >= 1:
            #             self.new_game = False
            #             self.bs = 0
            #             self.ws = 0
            #         self.update_game_view()
            #         for event in pg.event.get():
            #             if event.type == pg.QUIT:
            #                 done = True
            #             elif event.type == pg.VIDEORESIZE:
            #                 self.width, self.height = event.dict['size']
            #                 self.diameter = int(self.width / 800 * diameter)
            #                 self.button_size = int(self.width / 800 * button_size)
            #                 self.dot_size = int(self.width / 800 * dot_size)
            #
            #                 self.board_img_15 = pg.transform.smoothscale(board_img_15, (self.width, self.height))
            #                 self.stone_black = pg.transform.smoothscale(stone_black, (self.diameter, self.diameter))
            #                 self.stone_white = pg.transform.smoothscale(stone_white, (self.diameter, self.diameter))
            #                 self.mark_33 = pg.transform.smoothscale(mark_33, (self.diameter, self.diameter))
            #                 self.mark_44 = pg.transform.smoothscale(mark_44, (self.diameter, self.diameter))
            #                 self.mark_6 = pg.transform.smoothscale(mark_6, (self.diameter, self.diameter))
            #                 self.button_1 = pg.transform.smoothscale(button_1, (self.button_size, self.button_size))
            #                 self.button_2 = pg.transform.smoothscale(button_2, (self.button_size, self.button_size))
            #                 self.button_3 = pg.transform.smoothscale(button_3, (self.button_size, self.button_size))
            #                 self.button_4 = pg.transform.smoothscale(button_4, (self.button_size, self.button_size))
            #                 self.button_5 = pg.transform.smoothscale(button_5, (self.button_size, self.button_size))
            #                 self.button_6 = pg.transform.smoothscale(button_6, (self.button_size, self.button_size))
            #                 self.button_7 = pg.transform.smoothscale(button_7, (self.button_size, self.button_size))
            #                 self.update_game_view()
            #             elif event.type == pg.MOUSEBUTTONDOWN:
            #                 x, y = event.pos
            #                 row = round((y - 43 * self.width / 800) / (51 * self.width / 800))
            #                 col = round((x - 43 * self.width / 800) / (51 * self.width / 800))
            #
            #                 if 700 * self.width / 800 < x < 780 * self.width / 800 and y < 45 * self.width / 800:
            #                     if self.hint:
            #                         self.hint = False
            #                     else:
            #                         self.hint = True
            #                 elif 50 * self.width / 800 < x < 130 * self.width / 800 and y < 45 * self.width / 800:
            #                     if not self.new_game:
            #                         self.game.__init__()
            #                         self.bs = 0
            #                         self.ws = 0
            #                         self.new_game = True
            #                     else:
            #                         if self.bs == 0:
            #                             self.bs = 1
            #                         else:
            #                             self.ws = 1
            #                             self.new_game = False
            #                 elif 140 * self.width / 800 < x < 220 * self.width / 800 and y < 45 * self.width / 800:
            #                     if self.new_game:
            #                         if self.bs == 0:
            #                             self.bs = 2
            #                         else:
            #                             self.ws = 2
            #                             self.new_game = False
            #                 self.update_game_view()
        # pg.quit()

    # 보드 크기에 따라 놓고자 하는 돌의 row,col을 받아서 (x,y) 좌표 자동 조정
    def get_correct_stone_xy(self, row, col):
        board_size = self.board_arr.width
        rc = self.row_col_mid_const
        base_size = self.width_height_base
        ans = None
        if board_size == 15:
            ans = (round(self.width / 2 - (rc - col) * 51 * self.width / base_size - self.diameter / 2),
                   round(self.height / 2 - (rc - row) * 51 * self.height / base_size - self.diameter / 2))
        elif board_size == 13:
            ans = (round((43+col*51)-self.diameter/2),round((18+row*51)-self.diameter/2))
        elif board_size == 11:
            ans = (round((43 + col * 51) - self.diameter / 2), round((23 + row * 51) - self.diameter / 2))
        elif board_size == 9:
            ans = (round((43 + col * 51) - self.diameter / 2), round((20 + row * 51) - self.diameter / 2))
        else:
            print(f"구현되지 않은 판 크기 - {board_size}")
            quit()
        return ans

    def update_game_view(self):
        board = self.board_arr
        width_board = board.width  # GUI의 width X
        height_board = board.height  # GUI의 height X

        # screen = pg.display.set_mode((self.width, self.height + self.plus_height),pg.HWSURFACE | pg.DOUBLEBUF | pg.RESIZABLE)
        screen = pg.display.set_mode((self.width, self.height + self.plus_height),pg.HWSURFACE | pg.DOUBLEBUF)
        screen.blit(self.back_ground, (0, 0))
        screen.blit(self.get_board_img(), (0, 0))

        width_tmp = self.width
        height_tmp = self.height

        for row in range(height_board):
            for col in range(width_board):
                loc = row * width_board + col  # board의 states에서의 location
                stone_color = board.states.get(loc, -1)  # 보드 특정 좌표에서의 돌 색깔
                if stone_color == 1:  # 흑돌
                    screen.blit(self.stone_black, self.get_correct_stone_xy(row, col))
                    # screen.blit(self.stone_black,
                    #             (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                    #              round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))
                elif stone_color == 2:  # 백돌
                    screen.blit(self.stone_white, self.get_correct_stone_xy(row, col))
                    # screen.blit(self.stone_white,
                    #             (round(self.width / 2 - (7 - col) * 51 * self.width / width_tmp - self.diameter / 2),
                    #              round(self.height / 2 - (7 - row) * 51 * self.height / height_tmp - self.diameter / 2)))
                elif (row, col) in board.forbidden_locations:  # 금지 위치 (금수)일 경우, 화면에 표시해준다
                    index = board.forbidden_locations.index((row, col))
                    forbidden_type = board.forbidden_types[index]
                    # code20221004202931
                    if forbidden_type == error_const.BANNED_33:
                        screen.blit(self.mark_33, self.get_correct_stone_xy(row, col))
                        # screen.blit(self.mark_33,
                        #             (
                        #             round(self.width / 2 - (7 - col) * 51 * self.width / width_tmp - self.diameter / 2),
                        #             round(self.height / 2 - (
                        #                         7 - row) * 51 * self.height / height_tmp - self.diameter / 2)))
                    elif forbidden_type == error_const.BANNED_44:
                        screen.blit(self.mark_44, self.get_correct_stone_xy(row, col))
                        # screen.blit(self.mark_44,
                        #             (
                        #             round(self.width / 2 - (7 - col) * 51 * self.width / width_tmp - self.diameter / 2),
                        #             round(self.height / 2 - (
                        #                         7 - row) * 51 * self.height / height_tmp - self.diameter / 2)))
                    elif forbidden_type == error_const.BANNED_6:
                        screen.blit(self.mark_6, self.get_correct_stone_xy(row, col))
                        # screen.blit(self.mark_6,
                        #             (
                        #             round(self.width / 2 - (7 - col) * 51 * self.width / width_tmp - self.diameter / 2),
                        #             round(self.height / 2 - (
                        #                         7 - row) * 51 * self.height / height_tmp - self.diameter / 2)))

        if self.game_mode == 'replay':
            width_tmp = self.width
            height_tmp = self.height
            board_size = self.board_arr.width
            loc_prev = None
            loc_next = None
            if board_size == 15:
                loc_prev = (width_tmp / 4, height_tmp)
                loc_next = (width_tmp / 4 + 250, height_tmp)
            elif board_size == 13: # 671
                loc_prev = (130,height_tmp)
                loc_next = (390,height_tmp)
            elif board_size == 11: # 576
                loc_prev = (100,height_tmp)
                loc_next = (326,height_tmp)
            elif board_size == 9: # 470
                loc_prev = (60,height_tmp)
                loc_next = (260,height_tmp)
            screen.blit(self.img_prev, loc_prev)
            screen.blit(self.img_next, loc_next)
            # screen.blit(self.img_prev, (680 * self.width / 800, -37 * self.width / 800))

        # 가장 최근에 어디 놨는지 보여주는 기능
        if self.board_arr.last_loc != -1:
            row = self.board_arr.last_loc[0]
            col = self.board_arr.last_loc[1]
            if self.board_arr.current_player == 1:
                if self.game.board.order == 0:  # order = 0이면 사람이 먼저 시작 (사람이 흑)
                    color = (0, 0, 0)
                else:
                    color = (255, 255, 255)
            else:
                if self.game.board.order == 0:
                    color = (255, 255, 255)
                else:
                    color = (0, 0, 0)
            recent_rect = self.get_recent_rect(row,col)
            pg.draw.rect(screen, color, recent_rect, 2)
            # pg.draw.rect(screen, color, [round(self.width / 2 - (7 - col) * 51 * self.width / 800 - 20 / 2),
            #                              round(self.height / 2 - (7 - row) * 51 * self.height / 800 - 21 / 2), 21, 21], 2)

        # if self.hint:
        #     print("확률 설명은 구현중")
        #     # if self.game.state.check_turn():
        #     #     print("현재 흑이 이길 확률 : 구현중")
        #     #     # print('현재 흑이 이길 확률 :', round((float(get_value(self.model3, self.model4, self.game.state)) * 100 + 100) / 2, 2), '%')
        #     # else:
        #     #     print("현재 백이 이길 확률 : 구현중")
        #     #     # print('현재 백이 이길 확률 :', round((float(get_value(self.model3, self.model4, self.game.state)) * 100 + 100) / 2, 2), '%')
        #     # screen.blit(self.button_2, (680 * self.width / 800, -37 * self.width / 800))
        #     # p, m = get_policy(self.model, self.model2, self.game.state)
        #     # m = 1 / m
        #     # n = 0
        #     # for i in self.game.legal_actions:
        #     #     row = i // game.width
        #     #     col = i % game.width
        #     #     s.set_alpha(256 * p[n] * m)
        #     #     n += 1
        #     #     screen.blit(s,
        #     #                 (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.dot_size / 2),
        #     #                  round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.dot_size / 2)))
        # else:
        #     print("")
        #     # screen.blit(self.button_1, (680 * self.width / 800, -37 * self.width / 800))

        pg.display.flip()

    def action_replay(self, x, y):
        current_stone = 1 if self.replay_turn % 2 == 0 else 2  # 흑 또는 백
        start_x_1 = None
        start_x_2 = None
        if self.board_arr.width == 15:
            start_x_1 = 200
            start_x_2 = 450
        elif self.board_arr.width == 13:
            start_x_1 = 130
            start_x_2 = 390
        elif self.board_arr.width == 11:
            start_x_1 = 100
            start_x_2 = 326
        elif self.board_arr.width == 9:
            start_x_1 = 60
            start_x_2 = 260
        idx = self.replay_turn
        if start_x_1 <= x <= start_x_1+150 and y > self.height:  # 이전 버튼
            if idx == 0:
                return
            before_move = int(self.replay_data[idx - 2]) if idx >= 2 else None
            self.game.board.do_undo(last_move=int(self.replay_data[idx - 1]), before_move=before_move)
            self.replay_turn -= 1
            self.update_game_view()
        elif start_x_2 <= x <= start_x_2+150 and y > self.height:  # 다음 버튼
            if idx == len(self.replay_data) - 1:
                Tk().wm_withdraw()
                messagebox.showinfo('', '마지막입니다')
                return
            self.game.board.do_move(int(self.replay_data[idx]), current_stone)
            self.replay_turn += 1
            self.update_game_view()
        print(f"{x},{y}")

    def get_board_img(self):
        board_size = self.board_arr.width
        if board_size == 15:
            return board_img_15
        elif board_size == 13:
            return board_img_13
        elif board_size == 11:
            return board_img_11
        elif board_size == 9:
            return board_img_9
        else:
            print(f"잘못된 크기 : {board_size}")
            quit()

    def get_row_column(self, x, y):
        board_size = self.board_arr.width
        # 43 : 위에서 첫 격자까지 차이 and 맨 왼쪽에서 첫 격자까지 차이
        # 51 : 격자 사이 간격 (고정)
        row = None
        col = None
        if board_size == 15:
            row = round((y - 43 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
            col = round((x - 43 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
        elif board_size == 13:
            row = round((y - 18 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
            col = round((x - 43 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
        elif board_size == 11:
            row = round((y - 23 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
            col = round((x - 43 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
        elif board_size == 9:
            row = round((y - 20 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
            col = round((x - 43 * self.width / self.width_height_base) / (51 * self.width / self.width_height_base))
        else:
            print("없는 크기 - get_row_column")
            quit()
        return row,col

    def get_recent_rect(self, row, col):
        board_size = self.board_arr.width # 18 23 20 : 43
        if board_size == 15:
            return [round(self.width / 2 - (self.row_col_mid_const - col) * 51 * self.width / self.width_height_base - 20 / 2),
                    round(self.height / 2 - (self.row_col_mid_const - row) * 51 * self.height / self.width_height_base - 21 / 2),
                    21, 21]
        elif board_size == 13:
            return [round(self.width / 2 - (self.row_col_mid_const - col) * 51 * self.width / self.width_height_base + 5 / 2),
                    round(self.height / 2 - (self.row_col_mid_const - row) * 51 * self.height / self.width_height_base - 46 / 2),
                    21, 21]
        elif board_size == 11:
            return [round(self.width / 2 - (self.row_col_mid_const - col) * 51 * self.width / self.width_height_base + 1 / 2),
                    round(self.height / 2 - (self.row_col_mid_const - row) * 51 * self.height / self.width_height_base - 41 / 2),
                    21, 21]
        elif board_size == 9:
            return [round(self.width / 2 - (self.row_col_mid_const - col) * 51 * self.width / self.width_height_base + 1 / 2),
                    round(self.height / 2 - (self.row_col_mid_const - row) * 51 * self.height / self.width_height_base - 44 / 2),
                    21, 21]
        else:
            print("잘못된 크기 - get_recent_rect")
            quit()
        pass


if __name__ == '__main__':
    print("단독 실행이 불가능합니다")
    # gui = Gui()
    # gui.run()
