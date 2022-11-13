import ctypes

import pygame
import sys
from tkinter import messagebox
from tkinter import *
import gui_main

ctypes.windll.user32.SetProcessDPIAware()

# pygame.init() will initialize all
# imported module
pygame.init()

clock = pygame.time.Clock()

# it will display on screen
screen = pygame.display.set_mode([700, 400])
img_main = pygame.image.load('images/main_page.png')
# basic font for user typed
base_font = pygame.font.SysFont("notosanscjkkrregular", 32)
user_text_hard = ''
user_test_size = ''

# create rectangle
input_rect_hard = pygame.Rect(400, 170, 400, 40)
input_rect_size = pygame.Rect(400, 120, 400, 40)

# color_active stores color(lightskyblue3) which
# gets active when input box is clicked by user
color_active = pygame.Color('lightskyblue3')

# color_passive store color(chartreuse4) which is
# color of input box
color_passive = pygame.Color('chartreuse4')
color = color_passive

active_hard = False
active_size = False
status = 0  # stats가 0이면 난이도 선택 / status가 1이면 메인페이지 로딩


def load_gui_main():
    if not (user_text_hard.isdigit() or user_test_size.isdigit()):
        Tk().wm_withdraw()  # to hide the main window
        messagebox.showinfo('오류', '숫자만 입력해주세요')
    elif not 9 <= int(user_test_size) <= 15:
        Tk().wm_withdraw()  # to hide the main window
        messagebox.showinfo('오류', '판 크기는 9에서 15까지만 가능합니다')
    else:
        text_loading = base_font.render('모델을 로딩중입니다...', True, (0, 0, 0))
        screen = pygame.display.set_mode([700, 400])
        screen.blit(img_main, (0, 0))
        screen.blit(text_loading, (200,150))
        pygame.display.flip()
        gui = gui_main.Gui(ai_library='tensorflow', board_size=int(user_test_size), hard_gui=int(user_text_hard))
        gui.run()
        gui.update_game_view('main')
        is_run = True
        pygame.quit()
                # Tk().wm_withdraw()  # to hide the main window
                # messagebox.showinfo('오류', '존재하지 않는 난이도입니다')


while True:
    for event in pygame.event.get():

        # if user types QUIT then the screen will close
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if input_rect_hard.collidepoint(event.pos):
                active_hard = True
            else:
                active_hard = False

            if input_rect_size.collidepoint(event.pos):
                active_size = True
            else:
                active_size = False

        if event.type == pygame.KEYDOWN:
            print(event.key)
            # Check for backspace
            if event.key == pygame.K_BACKSPACE:
                if active_hard:
                    user_text_hard = user_text_hard[:-1]
                elif active_size:
                    user_test_size = user_test_size[:-1]
            elif event.key == 13:
                load_gui_main()
            # Unicode standard is used for string
            # formation
            else:
                if active_hard:
                    user_text_hard += event.unicode
                elif active_size:
                    user_test_size += event.unicode


    # it will set background color of screen
    screen.fill((255, 255, 255))
    pygame.transform.smoothscale(img_main,(600,500))
    screen.blit(img_main, (0, 0))

    # if active:
    #     color = color_active
    # else:
    #     color = color_passive

    # draw rectangle and argument passed which should
    # be on screen
    pygame.draw.rect(screen, color, input_rect_size)
    pygame.draw.rect(screen, color, input_rect_hard)

    text_surface_size = base_font.render(user_test_size, True, (255, 255, 255))
    text_surface_hard = base_font.render(user_text_hard, True, (255, 255, 255))

    # render at position stated in arguments
    screen.blit(text_surface_size, (input_rect_size.x + 5, input_rect_size.y))
    screen.blit(text_surface_hard, (input_rect_hard.x + 5, input_rect_hard.y))

    text_explain_size = base_font.render('판 크기 : ',True,(0,0,0))
    text_explain_hard = base_font.render(' 난이도 : ',True,(0,0,0))
    screen.blit(text_explain_size,(input_rect_size.x-150,input_rect_size.y-5))
    screen.blit(text_explain_hard,(input_rect_hard.x-150,input_rect_hard.y-5))

    # set width of textfield so that text cannot get
    # outside of user's text input
    input_rect_size.w = max(100, text_surface_size.get_width() + 10)
    input_rect_hard.w = max(100, text_surface_hard.get_width() + 10)

    # display.flip() will update only a portion of the
    # screen to updated, not full area
    pygame.display.flip()

    # clock.tick(60) means that for every second at most
    # 60 frames should be passed.
    clock.tick(60)