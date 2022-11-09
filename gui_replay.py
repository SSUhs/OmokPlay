import pygame as pg
import ctypes
import game
import time
from tkinter import messagebox
from tkinter import *

from constant import error_const


ctypes.windll.user32.SetProcessDPIAware()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
size = [800, 800]
diameter = 45
button_size = 120
dot_size = 12

img_background = pg.image.load('images/back_ground.png')


board_img = pg.image.load('images/오목판.png')
stone_black = pg.image.load('images/stone_black2.png')
stone_white = pg.image.load('images/stone_white.png')
mark_33 = pg.image.load('images/33.png')
mark_44 = pg.image.load('images/44.png')
mark_6 = pg.image.load('images/6.png')
img_next = pg.image.load('images/다음.png')
img_prev = pg.image.load('images/이전.png')
#icon = pg.image.load('images/icon5.png')
s = pg.Surface((12,12))
s.fill((255,0,0))
last = pg.Surface((20,20))

clock = pg.time.Clock()
#pg.display.set_icon(pg.transform.smoothscale(icon, (32, 32)))
pg.display.set_caption("리플레이")

# 오류 상수들
CONST_WRONG_POSITION = 1;

class Gui:
    def __init__(self, game, board_arr, player1, player2):
        # self.game_org = game.Game()
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.board_arr = board_arr
        self.width, self.height = 800, 850
        self.diameter = diameter
        self.button_size = button_size
        self.dot_size = dot_size
        self.board_img = board_img
        self.img_background = img_background
        self.stone_black = stone_black
        self.stone_white = stone_white
        self.mark_33 = mark_33
        self.mark_44 = mark_44
        self.mark_6 = mark_6
        self.img_next = img_next
        self.img_prev = img_prev
        self.hint = False
        self.new_game = False
        self.update_game_view()
        #
        # self.model = load_model('./model/policy_black.h5', compile=False)
        # self.model2 = load_model('./model/policy_white.h5', compile=False)
        # self.model3 = load_model('./model/value_black_t3.h5', compile=False)
        # self.model4 = load_model('./model/value_white_t3.h5', compile=False)

    def resize_view(self, event=None):
        if not event is None:
            self.width, self.height = event.dict['size']
        self.diameter = int(self.width / 800 * diameter)
        self.button_size = int(self.width / 800 * button_size)
        self.dot_size = int(self.width / 800 * dot_size)

        self.board_img = pg.transform.smoothscale(board_img, (self.width, self.height-50))
        self.img_background = pg.transform.smoothscale(img_background,(self.width,self.height))
        self.stone_black = pg.transform.smoothscale(stone_black, (self.diameter, self.diameter))
        self.stone_white = pg.transform.smoothscale(stone_white, (self.diameter, self.diameter))
        self.mark_33 = pg.transform.smoothscale(mark_33, (self.diameter, self.diameter))
        self.mark_44 = pg.transform.smoothscale(mark_44, (self.diameter, self.diameter))
        self.mark_6 = pg.transform.smoothscale(mark_6, (self.diameter, self.diameter))
        self.img_next = pg.transform.smoothscale(img_next,(self.button_size,50))  # 다음 버튼
        self.img_prev = pg.transform.smoothscale(img_prev,(self.button_size,50))  # 이전 버튼
        self.update_game_view()

    def run(self):
        done = False
        # self.game.init_play(self, self.player1, self.player2, is_shown=1)
        self.resize_view()
        self.update_game_view()
        # current_player = self.game.board.get_current_player() # 1은 사람, 2는 컴퓨터
        # if current_player == 2: # 사용자가 "백"이고 컴퓨터가 "흑"이면 컴퓨터가 먼저 놓기
        #     self.game.do_next(-1,-1)  # 컴퓨터 차례이므로 row, col 대입 X
        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                elif event.type == pg.VIDEORESIZE:
                    print("resize event")
                    self.resize_view(event)
                elif event.type == pg.MOUSEBUTTONDOWN:
                    x,y = event.pos
                    row = round((y - 43 * self.width / 800) / (51 * self.width / 800))
                    col = round((x - 43 * self.width / 800) / (51 * self.width / 800))
                    print("row, col : (",row,",",col,")")
        pg.quit()




    def update_game_view(self, player1 = None, player2 = None):
        screen = pg.display.set_mode((self.width, self.height), pg.HWSURFACE | pg.DOUBLEBUF | pg.RESIZABLE)
        screen.blit(self.img_background, (0,0))
        screen.blit(self.board_img, (0, 0))
        start_button_x = 260
        x_gap = 150
        screen.blit(self.img_next,(start_button_x,self.height-55))
        screen.blit(self.img_prev,(start_button_x+x_gap,self.height-55))
        if self.board_arr is None:
            print("테스트 모드 : update_gameview")
            pg.display.flip()
            return
        board = self.board_arr
        width_board = board.width # GUI의 width X
        height_board = board.height # GUI의 height X

        for row in range(height_board):
            for col in range(width_board):
                loc = row * width_board + col # board의 states에서의 location
                p = board.states.get(loc,-1)
                if p == player1:
                    if board.order == 0: # ●
                        screen.blit(self.stone_black,
                                    (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                                     round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))
                    else: # ○
                        screen.blit(self.stone_white,
                                    (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                                     round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))
                elif p == player2:
                    if board.order == 0: # ●
                        screen.blit(self.stone_white,
                                    (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                                     round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))
                    else: # ○
                        screen.blit(self.stone_black,
                                    (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                                     round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))
                elif (row,col) in board.forbidden_locations : # 금지 위치 (금수)일 경우, 화면에 표시해준다
                    index = board.forbidden_locations.index((row,col))
                    forbidden_type = board.forbidden_types[index]
                    if forbidden_type == error_const.BANNED_33:
                        print("3x3 가능성 발견 : 금지 이미지 blit")
                        screen.blit(self.mark_33,
                                    (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                                     round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))
                    elif forbidden_type == error_const.BANNED_44:
                        print("4x4 가능성 발견 : 금지 이미지 blit")
                        screen.blit(self.mark_44,
                                    (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                                     round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))
                    elif forbidden_type == error_const.BANNED_6:
                        print("6목 가능성 발견 : 금지 이미지 blit")
                        screen.blit(self.mark_6,
                                    (round(self.width / 2 - (7 - col) * 51 * self.width / 800 - self.diameter / 2),
                                     round(self.height / 2 - (7 - row) * 51 * self.height / 800 - self.diameter / 2)))


        # 가장 최근에 어디 놨는지 보여주는 기능
        if self.board_arr.last_loc != -1:
            row = self.board_arr.last_loc[0]
            col = self.board_arr.last_loc[1]
            if self.board_arr.current_player == 1:
                if self.game.board.order == 0:  # order = 0이면 사람이 먼저 시작 (사람이 흑)
                    color = (0,0,0)
                else:
                    color = (255,255,255)
            else:
                if self.game.board.order == 0:
                    color = (255,255,255)
                else:
                    color = (0,0,0)
            pg.draw.rect(screen, color, [round(self.width / 2 - (7 - col) * 51 * self.width / 800 - 20 / 2),
                                                       round(self.height / 2 - (7 - row) * 51 * self.height / 800 - 21 / 2),
                                                       21,
                                                       21], 2)

        pg.display.flip()


if __name__ == '__main__':
    # print("단독 실행이 불가능합니다")
    gui = Gui(None,None,None,None)
    gui.run()
