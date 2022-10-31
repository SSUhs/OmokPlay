import pygame as pg
import ctypes
from Human import Human
import game
import time
from tkinter import messagebox
from tkinter import *

import gui_ai_vs_player
from game import Board, Game
import pickle
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy

from constant import error_const
from policy_value_net_tensorflow import PolicyValueNetTensorflow

ctypes.windll.user32.SetProcessDPIAware()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
size = [800, 800]
diameter = 45
button_size = 240
dot_size = 12

img_main = pg.image.load('images/main_page.png')
img_newgame_black = pg.image.load('images/게임시작_흑.png')
img_newgame_white = pg.image.load('images/게임시작_백.png')
#icon = pg.image.load('images/icon5.png')
s = pg.Surface((12,12))
s.fill((255,0,0))
last = pg.Surface((20,20))

clock = pg.time.Clock()
#pg.display.set_icon(pg.transform.smoothscale(icon, (32, 32)))
pg.display.set_caption("오목")

class Gui:
    def __init__(self, ai_library, hard_gui):
        # self.game_org = game.Game()
        self.game = game
        self.ai_library = ai_library
        self.width, self.height = 800, 800
        self.diameter = diameter
        self.button_size = button_size
        self.dot_size = dot_size
        self.main_image = img_main
        self.img_newgame_black = img_newgame_black
        self.img_newgame_white = img_newgame_white
        self.x_bt_newgame_black = 0
        self.y_bt_newgame_black = 0
        self.x_bt_newgame_white = 0
        self.y_bt_newgame_white = 0
        self.bs = 0
        self.ws = 0
        self.hard_gui = hard_gui
        self.update_game_view('main')
        # self.model = load_model('./model/policy_black.h5', compile=False)
        # self.model2 = load_model('./model/policy_white.h5', compile=False)
        # self.model3 = load_model('./model/value_black_t3.h5', compile=False)
        # self.model4 = load_model('./model/value_white_t3.h5', compile=False)

    def resize_view(self, event=None, mode=None):
        if not event is None:
            self.width, self.height = event.dict['size']
        self.diameter = int(self.width / 800 * diameter)
        self.button_size = int(self.width / 800 * button_size)
        self.dot_size = int(self.width / 800 * dot_size)
        self.main_image = pg.transform.smoothscale(img_main, (self.width, self.height))
        self.img_newgame_black = pg.transform.smoothscale(img_newgame_black,(240,240))
        self.img_newgame_white = pg.transform.smoothscale(img_newgame_white,(240,240))
        # self.img_newgame = pg.transform.smoothscale(img_newgame, (self.button_size, self.button_size))
        self.update_game_view(mode)


    def load_game(self, black_white):
        # print(black_white)
        hard_gui = self.hard_gui
        num = 5
        width, height = 9, 9

        if self.ai_library == 'theano':
            model_file = './model/policy_9_' + str(hard_gui) + ".model"
            gui_board = None
            board_arr = Board(width=width, height=height, n_in_row=num)
            game = Game(board_arr, is_gui_mode=True)
            if black_white == 'black':
                order = 0
            elif black_white == 'white':
                order = 1
            else:
                print("없는 모드입니다")
                pg.quit()  # 종료

            # 이미 학습된 model을 불러와서 학습된 policy_value_net을 얻는다
            policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
            best_policy = PolicyValueNetNumpy(width, height, policy_param)

            # n_playout값 : 성능
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
            human = Human()

            game.board.init_board(start_player=order)
            gui_board = gui_ai_vs_player.Gui(game, board_arr, human, mcts_player)
            gui_board.run()
            gui_board.update_game_view()
            pg.quit()
        elif self.ai_library == 'tensorflow':  # 텐서플로우 학습 모델 기반으로 게임 시작
            model_file =  f'./model/tf_policy_{width}_{str(hard_gui)}_model'
            gui_board = None
            board_arr = Board(width=width, height=height, n_in_row=num)
            game = Game(board_arr, is_gui_mode=True)
            if black_white == 'black':
                order = 0
            elif black_white == 'white':
                order = 1
            else:
                print("없는 모드입니다")
                pg.quit()  # 종료
            # 이미 학습된 model을 불러와서 학습된 policy_value_net을 얻는다
            best_policy = PolicyValueNetTensorflow(width, height,model_file,compile_env='local')  # 코랩에서는 start_game.py 수행 안하기 때문에 compile_env는 local로 고정
            mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5,
                                     n_playout=400)  # set larger n_playout for better performance
            human = Human()

            game.board.init_board(start_player=order)
            gui_board = gui_ai_vs_player.Gui(game, board_arr, human, mcts_player)
            gui_board.run()
            gui_board.update_game_view()
            pg.quit()

        else:
            print("지원 되지 않는 라이브러리입니다")
            quit()




    def run(self):
        done = False
        self.resize_view(None,'main')
        # self.game.init_play(self, human, mcts_player, is_shown=1)

        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                elif event.type == pg.VIDEORESIZE:
                    self.resize_view(event,'main')
                elif event.type == pg.MOUSEBUTTONDOWN:
                    rect_black = self.img_newgame_black.get_rect()
                    rect_white = self.img_newgame_white.get_rect()
                    x,y = event.pos
                    # print("x2 : ",x2)
                    # print("y2 : ",y2)
                    # new_game 버튼 내부에 있을 경우
                    if x > self.x_bt_newgame_black + 51 and x < self.x_bt_newgame_black + rect_black.width - 51 and y > self.y_bt_newgame_black + 82 and y < self.y_bt_newgame_black + rect_black.height -82:
                        self.load_game('black')
                    elif x > self.x_bt_newgame_white + 51 and x < self.x_bt_newgame_white + rect_white.width - 51 and y > self.y_bt_newgame_white + 82 and y < self.y_bt_newgame_white + rect_white.height -82:
                        self.load_game('white')



        pg.quit()



    def update_game_view(self, mode):
        screen = pg.display.set_mode((self.width, self.height), pg.HWSURFACE | pg.DOUBLEBUF | pg.RESIZABLE)
        screen.blit(self.main_image, (0, 0))
        if mode == 'main':
            # x = (self.width-self.button_size)/2
            # y = 100
            self.x_bt_newgame_black = (self.width-self.button_size)/2
            self.x_bt_newgame_white = (self.width-self.button_size)/2
            self.y_bt_newgame_black = 0
            self.y_bt_newgame_white = 100
            # x = 680 * self.width / 800
            # y = -37 * self.width / 800
            # print("놓은 자리 :",x,y)
            screen.blit(self.img_newgame_black,(self.x_bt_newgame_black,self.y_bt_newgame_black))
            screen.blit(self.img_newgame_white,(self.x_bt_newgame_white,self.y_bt_newgame_white))
            # screen.blit(self.img_newgame, (x,y))
            # screen.blit(self.img_newgame, (680 * self.width / 800, -37 * self.width / 800))

        pg.display.flip()


if __name__ == '__main__':
    print("start_game.py로 실행해주세요")
    quit()
#     gui.run()
