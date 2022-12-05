import configparser
import os

# 기본 값
config = configparser.ConfigParser()


def load_setting_file():
    if os.path.isfile('setting/config.ini'):
        make_new_config()


def read_config(key):
    # 설정파일 읽기
    config = configparser.ConfigParser()
    config.read('setting/config.ini', encoding='utf-8')
    # 설장파일 색션 확인
    config.sections()
    # 섹션값 읽기
    if key == 'board_size':
        board_size = int(config['option']['board_size'])
        return board_size
    elif key == 'ai_hard':
        ai_hard = int(config['option']['ai_hard'])
        return ai_hard
    elif key == 'use_mcts':
        use_mcts = bool(config['option']['use_mcts'])
        return use_mcts
    elif key == 'mcts_playout':
        mcts_playout = int(config['option']['mcts_playout'])
        return mcts_playout
    else:
        print("존재하지 않는 키")



def make_new_config():
    config['option'] = {}
    config['option']['board_size'] = '15'
    config['option']['ai_hard'] = '4'  # 4,3,2,1
    config['option']['use_mcts'] = 'True'
    config['option']['mcts_playout'] = '100'

def save_config():
    # 설정파일 저장
    with open('setting/config.ini', 'w', encoding='utf-8') as configfile:
        config.write(configfile)

