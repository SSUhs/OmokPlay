# 프로젝트랑 관련 X

import pickle

my_list = ['a','b','c']
with open("test_pickle.pickle","wb") as fw:
        pickle.dump(my_list,fw)

with open("test_pickle.pickle","rb") as fr:
    data = pickle.load(fr)

print(data)