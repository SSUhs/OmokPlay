import pygame as pg
import ctypes

import csv
import player_AI
from Human import Human
import game
from game import Board, Game
import pickle
from setting import setting
from tkinter import *
# from PIL import ImageTk, Image
from tkinter import filedialog
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

ctypes.windll.user32.SetProcessDPIAware()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
size = [800, 800]
diameter = 45
button_width = 300
button_height = 80
dot_size = 12
show_frame = False  # True면 hide를 풀고 show_frame은 다시 False로

img_main = pg.image.load('images/theme_white_gray/main_page.png')  # img_main = pg.image.load('images/main_page.png')
img_ai_vs_player = pg.image.load(
    'images/theme_white_gray/AI_VS_플레이어.png')  # img_ai_vs_player = pg.image.load('images/시작(흑).png')
img_player2 = pg.image.load('images/theme_white_gray/2인 플레이.png')
img_replay = pg.image.load('images/theme_white_gray/리플레이.png')
img_player_black = pg.image.load('images/theme_white_gray/플레이어-흑.png')
img_player_white = pg.image.load('images/theme_white_gray/플레이어-백.png')
img_back = pg.image.load('images/theme_white_gray/뒤로가기.png')
img_upper = pg.image.load('images/theme_white_gray/상_버튼.png')
img_middle = pg.image.load('images/theme_white_gray/중_버튼.png')
img_lower = pg.image.load('images/theme_white_gray/하_버튼.png')
img_setting = pg.image.load('images/theme_white_gray/판 크기 설정.png')
img_cur_15x15 = pg.image.load('images/theme_white_gray/current_size_15.png')
img_cur_13x13 = pg.image.load('images/theme_white_gray/current_size_13.png')
img_cur_11x11 = pg.image.load('images/theme_white_gray/current_size_11.png')
img_cur_9x9 = pg.image.load('images/theme_white_gray/current_size_9.png')
img_button_15 = pg.image.load('images/theme_white_gray/button_15.png')
img_button_13 = pg.image.load('images/theme_white_gray/button_13.png')
img_button_11 = pg.image.load('images/theme_white_gray/button_11.png')
img_button_9 = pg.image.load('images/theme_white_gray/button_9.png')


# img_newgame_white = pg.image.load('images/theme_white_gray/게임 시작 (백).png') # img_newgame_white = pg.image.load('images/시작(백).png')


# icon = pg.image.load('images/icon5.png')
s = pg.Surface((12, 12))
s.fill((255, 0, 0))
last = pg.Surface((20, 20))

clock = pg.time.Clock()
# pg.display.set_icon(pg.transform.smoothscale(icon, (32, 32)))
pg.display.set_caption("오목")


def get_play_data(csv_file_path):
    with open(csv_file_path, 'r') as f:
        # next(f, None)
        reader = csv.reader(f)
        list_all = []
        for row in reader:
            list_all.append(row)
    try:
        game_mode_csv = list_all[0][1]
        board_size_csv = int(list_all[1][1])
        winner_csv = list_all[2][1]
        game_state_list = list_all[3]
    except:
        return None, None, None, None

    return game_mode_csv, board_size_csv, winner_csv, game_state_list


class Gui:
    def __init__(self):
        # self.game_org = game.Game()
        self.width_height = None
        self.game = game
        self.width, self.height = 800, 800
        self.diameter = diameter
        self.dot_size = dot_size
        self.button_size = button_width
        self.main_image = img_main
        self.img_ai_vs_player = img_ai_vs_player
        self.img_replay = img_replay
        self.img_player2 = img_player2
        self.img_setting = img_setting
        self.img_15x15 = img_button_15
        self.img_13x13 = img_button_13
        self.img_11x11 = img_button_11
        self.img_9x9 = img_button_9
        self.img_cur_15x15 = img_cur_15x15
        self.img_cur_13x13 = img_cur_13x13
        self.img_cur_11x11 = img_cur_11x11
        self.img_cur_9x9 = img_cur_9x9
        self.img_upper = img_upper
        self.img_middle = img_middle
        self.img_lower = img_lower
        self.img_player_black = img_player_black
        self.img_player_white = img_player_white
        self.img_back = img_back
        self.is_train_set_mode = None
        self.is_human_intervene = None
        self.best_policy = None  # 알파고 제로 기반의 경우 여기에 value망까지 포함
        self.best_value = None
        self.use_mcts = None
        self.ai_library = None
        self.hard_gui = None
        self.is_test_mode = None
        self.bs = 0
        self.ws = 0
        self.current_frame = 'main'  # main / setting / select_stone
        self.load_setting()
        self.update_game_view()

        # self.model = load_model('./model/policy_black.h5', compile=False)
        # self.model2 = load_model('./model/policy_white.h5', compile=False)
        # self.model3 = load_model('./model/value_black_t3.h5', compile=False)
        # self.model4 = load_model('./model/value_white_t3.h5', compile=False)

    def get_img_current_size(self):
        img_current = None
        if self.width_height == 15:
            img_current = self.img_cur_15x15
        elif self.width_height == 13:
            img_current = self.img_cur_13x13
        elif self.width_height == 11:
            img_current = self.img_cur_11x11
        elif self.width_height == 9:
            img_current = self.img_cur_9x9
        else:
            print("잘못된 크기")
            quit()
        return img_current




    # 설정 변경
    def load_setting(self):
        setting.load_setting_file()
        ai_library = 'tensorflow'
        is_test_mode = False
        board_size = int(setting.read_config('board_size'))
        self.change_board_size(board_size)
        # hard = setting.read_config('ai_hard')
        # is_human_intervene = None
        # use_mcts = None
        # if hard == 4 or hard == 3:
        #     use_mcts = True
        #     is_human_intervene = True
        # elif hard == 3:
        #     use_mcts = False
        #     is_human_intervene = True
        # elif hard == 2:
        #     use_mcts = False
        #     is_human_intervene = False
        # else:
        #     print("없는 난이도")
        #     quit()
        # self.use_mcts = use_mcts
        # self.hard_gui = hard
        self.ai_library = ai_library
        self.is_test_mode = is_test_mode
        # self.is_train_set_mode = is_train_set_mode
        # self.is_human_intervene = is_human_intervene


    def resize_view(self, event=None):
        if not event is None:
            self.width, self.height = event.dict['size']
        self.diameter = int(self.width / 800 * diameter)
        self.button_size = int(self.width / 800 * button_width)
        self.dot_size = int(self.width / 800 * dot_size)
        self.main_image = pg.transform.smoothscale(img_main, (self.width, self.height))
        self.img_ai_vs_player = pg.transform.smoothscale(img_ai_vs_player, (button_width, button_height))  # 169, 87
        self.img_player2 = pg.transform.smoothscale(img_player2, (button_width, button_height))
        self.img_replay = pg.transform.smoothscale(img_replay, (button_width, button_height))
        self.img_setting = pg.transform.smoothscale(img_setting, (button_width, button_height))
        self.img_15x15 = pg.transform.smoothscale(img_button_15, (button_width, button_height))
        self.img_13x13 = pg.transform.smoothscale(img_button_13, (button_width, button_height))
        self.img_11x11 = pg.transform.smoothscale(img_button_11, (button_width, button_height))
        self.img_9x9 = pg.transform.smoothscale(img_button_9, (button_width, button_height))
        self.img_cur_15x15 = pg.transform.smoothscale(img_cur_15x15, (button_width, button_height))
        self.img_cur_13x13 = pg.transform.smoothscale(img_cur_13x13, (button_width, button_height))
        self.img_cur_11x11 = pg.transform.smoothscale(img_cur_11x11, (button_width, button_height))
        self.img_cur_9x9 = pg.transform.smoothscale(img_cur_9x9, (button_width, button_height))
        self.img_player_black = pg.transform.smoothscale(img_player_black, (button_width, button_height))
        self.img_player_white = pg.transform.smoothscale(img_player_white, (button_width, button_height))
        self.img_back = pg.transform.smoothscale(img_back, (button_width, button_height))
        self.img_upper = pg.transform.smoothscale(img_upper, (button_width, button_height))
        self.img_middle = pg.transform.smoothscale(img_middle, (button_width, button_height))
        self.img_lower = pg.transform.smoothscale(img_lower, (button_width, button_height))
        # self.img_newgame = pg.transform.smoothscale(img_newgame, (self.button_size, self.button_size))
        self.update_game_view()

    # game_mode : ai_vs_player / player_vs_player / replay
    # black_white_human : AI랑 대결시 사람의 흑, 돌
    def load_game(self, black_white_human=None, game_mode=None):
        import gui_play_game
        # is_human_intervene = int(input("사람 알고리즘 개입 하는 경우 1, 아닌 경우 0 입력 : "))
        # if is_human_intervene == 0:
        #     is_human_intervene = False
        # else:
        #     is_human_intervene = True
        # print(black_white)
        hard_gui = self.hard_gui
        num = 5
        order = None
        black_white_ai = None
        gui_board = None
        if game_mode == 'replay':
            csv_file_path = self.load_saved_game_csv()
            if csv_file_path is None or csv_file_path == '':  # 선택 안한 경우
                return
            mode_csv, board_size_csv, winner_csv, game_list = get_play_data(csv_file_path)
            board_arr = Board(width=board_size_csv, height=board_size_csv, n_in_row=num)
            game = Game(board_arr, is_gui_mode=True, game_mode='replay', record_list=game_list)
            order = 0
            human1 = Human()
            human2 = Human()
            game.board.init_board(start_player=order)
            gui_board = gui_play_game.Gui(game, board_arr, human1, human2, game_mode=game_mode,
                                          is_test_mode=self.is_test_mode,replay_data=game_list)
            gui_board.run()
            gui_board.update_game_view()
            screen = pg.display.set_mode((self.width, self.height), pg.HIDDEN)
            pg.display.flip()

        elif game_mode == 'player_vs_player':
            board_arr = Board(width=self.width_height, height=self.width_height, n_in_row=num,
                              is_train_set_mode=self.is_train_set_mode)
            game = Game(board_arr, is_gui_mode=True, is_human_intervene=self.is_human_intervene,
                        game_mode='player_vs_player')
            order = 0
            human1 = Human()
            human2 = Human()
            game.board.init_board(start_player=order)
            gui_board = gui_play_game.Gui(game, board_arr, human1, human2, game_mode=game_mode,
                                          is_test_mode=self.is_test_mode, black_white_ai=None)
            gui_board.run()
            gui_board.update_game_view()
            screen = pg.display.set_mode((self.width, self.height), pg.HIDDEN)
            pg.display.flip()
            # pg.quit()
        elif game_mode == 'ai_vs_player':
            if self.ai_library == 'tensorflow':  # 텐서플로우 학습 모델 기반으로 게임 시작
                board_arr = Board(width=self.width_height, height=self.width_height, n_in_row=num,
                                  is_train_set_mode=self.is_train_set_mode)
                game = Game(board_arr, is_gui_mode=True, is_human_intervene=self.is_human_intervene,
                            game_mode='ai_vs_player')
                if black_white_human == 'black':
                    black_white_ai = 'white'
                else:
                    black_white_ai = 'black'
                if black_white_human == 'black':
                    order = 0
                elif black_white_human == 'white':
                    order = 1
                else:
                    print("없는 모드입니다")
                    quit()
                if self.is_train_set_mode:
                    from player_AI import MCTSPlayer_TrainSet
                    computer_player = MCTSPlayer_TrainSet(self.best_policy, self.best_value, c_puct=5, n_playout=100,
                                                          is_selfplay=False, is_test_mode=self.is_test_mode,
                                                          is_human_intervene=self.is_human_intervene,
                                                          black_white_ai=black_white_ai, use_mcts=self.use_mcts)
                    human = Human()
                    game.board.init_board(start_player=order)
                    gui_board = gui_play_game.Gui(game, board_arr, human, computer_player,
                                                  is_test_mode=self.is_test_mode, black_white_ai=black_white_ai,
                                                  game_mode=game_mode)
                    gui_board.run()
                    gui_board.update_game_view()
                    # pg.quit()
                else:
                    from mcts_alphaZero import MCTSPlayer
                    board_arr = Board(width=self.width_height, height=self.width_height, n_in_row=num,
                                      is_train_set_mode=self.is_train_set_mode)
                    game = Game(board_arr, is_gui_mode=True, is_human_intervene=self.is_human_intervene,
                                game_mode='ai_vs_player')
                    computer_player = MCTSPlayer(self.best_policy.policy_value_fn, c_puct=5, n_playout=400)
                    human = Human()
                    game.board.init_board(start_player=order)
                    gui_board = gui_play_game.Gui(game, board_arr, human, computer_player, game_mode=game_mode,black_white_ai=black_white_ai)
                    gui_board.run()
                    gui_board.update_game_view()
                    screen = pg.display.set_mode((self.width, self.height), pg.HIDDEN)
                    pg.display.flip()

            elif self.ai_library == 'theano':
                from mcts_alphaZero import MCTSPlayer
                gui_board = None
                board_arr = Board(width=self.width_height, height=self.width_height, n_in_row=num,
                                  is_train_set_mode=self.is_train_set_mode)
                game = Game(board_arr, is_gui_mode=True, game_mode=game_mode)
                if black_white_human == 'black':
                    order = 0
                elif black_white_human == 'white':
                    order = 1
                else:
                    print("없는 모드입니다")
                    quit()
                    # pg.quit()  # 종료
                # n_playout값 : 성능
                computer_player = MCTSPlayer(self.best_policy.policy_value_fn, c_puct=5, n_playout=400)
                human = Human()
                import gui_play_game
                game.board.init_board(start_player=order)
                gui_board = gui_play_game.Gui(game, board_arr, human, computer_player, game_mode=game_mode)
                gui_board.run()
                gui_board.update_game_view()
                screen = pg.display.set_mode((self.width, self.height), pg.HIDDEN)
                pg.display.flip()
                # pg.quit()
            else:
                print("지원 되지 않는 라이브러리입니다")
                quit()
        else:
            print(f"{game_mode}는 존재하지 않는 모드입니다")
            quit()

    def run(self):
        done = False
        self.resize_view(None)
        global show_frame
        while not done:
            for event in pg.event.get():
                if show_frame:  # 화면 보이는 명령
                    self.update_game_view()
                    show_frame = False
                    continue
                if event.type == pg.QUIT:
                    pg.quit()
                elif event.type == pg.VIDEORESIZE:
                    self.resize_view(event)
                elif event.type == pg.MOUSEBUTTONDOWN:  # 액션
                    x, y = event.pos
                    print(f'x : {x} y : {y}')
                    self.action_mouse_click(x, y)

        pg.quit()

    def update_game_view(self):
        mode = self.current_frame
        # screen = pg.display.set_mode((self.width, self.height), pg.HWSURFACE | pg.DOUBLEBUF | pg.RESIZABLE)
        screen = pg.display.set_mode((self.width, self.height), pg.HWSURFACE | pg.DOUBLEBUF)
        screen.blit(self.main_image, (0, 0))
        if mode == 'main':
            x_start = (self.width - self.button_size) / 2
            y_lb_cur_size = 30
            y_bt_ai_vs_player = 150
            y_bt_player2 = 250
            y_bt_replay = 350
            y_bt_setting = 450
            screen.blit(self.get_img_current_size(),(x_start,y_lb_cur_size))
            screen.blit(self.img_ai_vs_player, (x_start, y_bt_ai_vs_player))
            screen.blit(self.img_replay, (x_start, y_bt_replay))
            screen.blit(self.img_player2, (x_start, y_bt_player2))
            screen.blit(self.img_setting,(x_start,y_bt_setting))
        elif mode == 'select_hard':
            x_start = (self.width - self.button_size) / 2
            y_start = 150
            screen.blit(self.img_upper, (x_start, y_start))
            screen.blit(self.img_middle, (x_start, y_start + 100))
            screen.blit(self.img_lower, (x_start, y_start + 200))
            screen.blit(self.img_back, (x_start, y_start + 300))
        elif mode == 'select_stone':
            x_start = (self.width - self.button_size) / 2
            y_start = 150
            screen.blit(self.img_player_black, (x_start, y_start))
            screen.blit(self.img_player_white, (x_start, y_start + 100))
            screen.blit(self.img_back, (x_start, y_start + 200))
        elif mode == 'setting':
            x_start = (self.width - self.button_size) / 2
            y_start = 150
            screen.blit(self.img_15x15, (x_start, y_start))
            screen.blit(self.img_13x13, (x_start, y_start + 100))
            screen.blit(self.img_11x11, (x_start, y_start + 200))
            screen.blit(self.img_9x9, (x_start, y_start + 300))
            screen.blit(self.img_back, (x_start, y_start + 400))
        else:
            print("없는 모드")
            quit()

        pg.display.flip()


    def load_model(self, is_reload):
        # if not is_reload and self.best_policy is not None: # 재로딩도 아니고 이미 모델이 로딩된 경우
        #     print("이미 load된 모델 존재")
        #     return

        hard_gui = self.hard_gui
        if self.ai_library == 'tensorflow' and not self.is_train_set_mode:
            from policy_value_net.policy_value_net_tensorflow import PolicyValueNetTensorflow
            model_file = f'./model/tf_policy_{self.width_height}_{self.hard_gui}_model'
            self.best_policy = PolicyValueNetTensorflow(self.width_height, self.width_height, model_file,
                                                        compile_env='local',init_num=hard_gui)
        elif self.ai_library == 'tensorflow' and self.is_train_set_mode:
            self.best_policy = player_AI.load_model_trainset_mode(model_type='policy', size=self.width_height,
                                                                  train_num=self.hard_gui)
            self.best_value = player_AI.load_model_trainset_mode(model_type='value', size=self.width_height,
                                                                 train_num=self.hard_gui)
        elif self.ai_library == 'theano':
            model_file = './model/policy_9_' + str(hard_gui) + ".model"
            # 이미 학습된 model을 불러와서 학습된 policy_value_net을 얻는다
            policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
            from policy_value_net.policy_value_net_numpy import PolicyValueNetNumpy
            best_policy = PolicyValueNetNumpy(self.width_height, self.width_height, policy_param)
            self.best_policy = best_policy
        else:
            print("존재하지 않는 경우")
            quit()

    def action_mouse_click(self, x, y):
        start_x = 318
        end_x = 480
        if self.current_frame == 'main':
            if not start_x <= x <= end_x:
                return
            if 153 <= y <= 233:
                self.current_frame = 'select_hard'
                self.update_game_view()
            elif 253 <= y <= 333:
                self.load_game(game_mode='player_vs_player')
                self.update_game_view()
            elif 353 <= y <= 433:
                self.load_game(game_mode='replay')
                self.update_game_view()
            elif 453 <= y <= 533:
                self.current_frame = 'setting'
                self.update_game_view()
        elif self.current_frame == 'select_hard':
            if not start_x <= x <= end_x:
                return
            if 153 <= y <= 233:  # 상
                self.change_hard('상')
                self.current_frame = 'select_stone'
                self.update_game_view()
            elif 253 <= y <= 333:  # 중
                self.change_hard('중')
                self.current_frame = 'select_stone'
                self.update_game_view()
            elif 353 <= y <= 433:  # 하
                self.change_hard('하')
                self.current_frame = 'select_stone'
                self.update_game_view()
            elif 453 <= y <= 533:  # 뒤로가기
                self.current_frame = 'main'
                self.update_game_view()
        elif self.current_frame == 'select_stone':
            if not start_x <= x <= end_x:
                return
            if 153 <= y <= 233:
                self.load_game('black', game_mode='ai_vs_player')
                self.update_game_view()
            elif 253 <= y <= 333:
                self.load_game('white', game_mode='ai_vs_player')
                self.update_game_view()
            elif 353 <= y <= 433:  # 뒤로가기
                self.current_frame = 'select_hard'
                self.update_game_view()
        elif self.current_frame == 'setting':
            if not start_x <= x <= end_x:
                return
            if 153 <= y <= 233:
                self.change_setting('size',15)
                self.current_frame = 'main'
                self.update_game_view()
            elif 253 <= y <= 333:
                self.change_setting('size',13)
                self.current_frame = 'main'
                self.update_game_view()
            elif 353 <= y <= 433:
                self.change_setting('size',11)
                self.current_frame = 'main'
                self.update_game_view()
            elif 453 <= y <= 533:
                self.change_setting('size',9)
                self.current_frame = 'main'
                self.update_game_view()
            elif 553 <= y <= 633: # 뒤로가기
                self.current_frame = 'main'
                self.update_game_view()
    def load_saved_game_csv(self):
        Tk().wm_withdraw()  # to hide the main window
        filename = filedialog.askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))
        print(filename)
        return filename


    def change_setting(self,type,arg):
        if type == 'size':
            setting.save_config(arg)
            self.change_board_size(arg)
        else:
            print("없는 경우")
            quit()
        return

    def change_board_size(self, board_size):
        board_size = int(board_size)
        self.width_height = board_size
        self.is_train_set_mode = True if board_size == 15 else False

    def change_hard(self,hard):
        if self.is_train_set_mode and self.ai_library == 'tensorflow' and self.width_height == 15:
            if hard == '상':
                self.is_human_intervene = True
                self.use_mcts = True
            elif hard == '중':
                self.is_human_intervene = True
                self.use_mcts = False
            elif hard == '하':
                self.is_human_intervene = False
                self.use_mcts = False
            else:
                print("없는 난이도 - change_hard")
                quit()
            self.load_model(True)
        elif self.ai_library == 'tensorflow' and self.width_height == 13:
            if hard == '상':
                self.hard_gui = 18670
            elif hard == '중':
                self.hard_gui = 14990
            elif hard == '하':
                self.hard_gui = 12010
            else:
                print("없는 난이도 - change_hard")
                quit()
            self.load_model(True)
        elif self.ai_library == 'tensorflow' and self.width_height == 11:
            if hard == '상':
                self.hard_gui = 13140
            elif hard == '중':
                self.hard_gui = 10260
            elif hard == '하':
                self.hard_gui = 6240
            else:
                print("없는 난이도 - change_hard")
                quit()
            self.load_model(True)
        elif self.ai_library == 'tensorflow' and self.width_height == 9:
            if hard == '상':
                self.hard_gui = 11440
            elif hard == '중':
                self.hard_gui = 6680
            elif hard == '하':
                self.hard_gui = 3320
            else:
                print("없는 난이도 - change_hard")
                quit()
            self.load_model(True)
        else:
            print("없는 크기 - change_hard")
            quit()


if __name__ == '__main__':
    gui = Gui()
    gui.run()
    gui.update_game_view()
    quit()
#     gui.run()
