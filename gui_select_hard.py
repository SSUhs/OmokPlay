import pygame
import sys
from tkinter import messagebox
from tkinter import *
import gui_main


# pygame.init() will initialize all
# imported module
pygame.init()

clock = pygame.time.Clock()

# it will display on screen
screen = pygame.display.set_mode([700, 400])
img_main = pygame.image.load('images/main_page.png')
# basic font for user typed
base_font = pygame.font.SysFont("notosanscjkkrregular", 32)
user_text = ''

# create rectangle
input_rect = pygame.Rect(400, 170, 400, 40)

# color_active stores color(lightskyblue3) which
# gets active when input box is clicked by user
color_active = pygame.Color('lightskyblue3')

# color_passive store color(chartreuse4) which is
# color of input box.
color_passive = pygame.Color('chartreuse4')
color = color_passive

active = False


def load_gui_main():
    if not user_text.isdigit():
        Tk().wm_withdraw()  # to hide the main window
        messagebox.showinfo('오류', '숫자만 입력해주세요')
    else:
        text_loading = base_font.render('모델을 로딩중입니다...', True, (0, 0, 0))
        screen = pygame.display.set_mode([700, 400])
        screen.blit(img_main, (0, 0))
        screen.blit(text_loading, (200,200))
        pygame.display.flip()
        is_run = False
        try:
            gui = gui_main.Gui(ai_library='tensorflow',hard_gui=int(user_text))
            gui.run()
            is_run = True
            pygame.quit()
        except:
            if not is_run:  # 모델 로딩 실패한경우
                Tk().wm_withdraw()  # to hide the main window
                messagebox.showinfo('오류', '존재하지 않는 난이도입니다')
            quit()


while True:
    for event in pygame.event.get():

        # if user types QUIT then the screen will close
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:

            if input_rect.collidepoint(event.pos):
                active = True
            else:
                active = False

        if event.type == pygame.KEYDOWN:
            print(event.key)
            # Check for backspace
            if event.key == pygame.K_BACKSPACE:

                # get text input from 0 to -1 i.e. end.
                user_text = user_text[:-1]
            elif event.key == 13:
                load_gui_main()
            # Unicode standard is used for string
            # formation
            else:
                user_text += event.unicode

    # it will set background color of screen
    screen.fill((255, 255, 255))
    pygame.transform.smoothscale(img_main,(600,500))
    screen.blit(img_main, (0, 0))

    if active:
        color = color_active
    else:
        color = color_passive

    # draw rectangle and argument passed which should
    # be on screen
    pygame.draw.rect(screen, color, input_rect)

    text_surface = base_font.render(user_text, True, (255, 255, 255))

    # render at position stated in arguments
    screen.blit(text_surface, (input_rect.x + 5, input_rect.y))

    text_explain = base_font.render('난이도를 입력하세요 : ',True,(0,0,0))
    screen.blit(text_explain,(input_rect.x-300,input_rect.y-5))

    # set width of textfield so that text cannot get
    # outside of user's text input
    input_rect.w = max(100, text_surface.get_width() + 10)

    # display.flip() will update only a portion of the
    # screen to updated, not full area
    pygame.display.flip()

    # clock.tick(60) means that for every second at most
    # 60 frames should be passed.
    clock.tick(60)