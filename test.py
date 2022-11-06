# # 프로젝트랑 관련 X
#
#

import pickle

from pandas import DataFrame

if __name__ == '__main__':
    with open('C:\\Users\\vvpas\\Desktop\\기본\\대학\\오픈소스기반기초설계\\프로젝트\\코랩가능\\tf_train_9_2590.pickle', 'rb') as file:
        try:
            data = pickle.load(file)
        except: print("zz")
    print("asdf")


