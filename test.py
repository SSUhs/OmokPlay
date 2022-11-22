# import numpy as np
# import csv
# import keras.backend as K
#
# def reshape_to_15_15_1(data):
#     return K.reshape(data, [-1, 15, 15, 1])
#
#
# def get_dataset(csv_name,is_one_hot_encoding,pv_type,type_train):
#     csv_name = csv_name
#     if pv_type == 'seperate':
#         data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v= tttttt(csv_name, is_one_hot_encoding=is_one_hot_encoding,type_train=type_train)
#     else:
#         print("미구현")
#     print("데이터 로딩 성공")
#     if type_train == 0:
#         data_x_p_black = reshape_to_15_15_1(data_x_p_black)
#     elif type_train == 1:
#         data_x_p_white = reshape_to_15_15_1(data_x_p_white)
#     elif type_train == 2:
#         data_x_v = reshape_to_15_15_1(data_x_v)
#     else:
#         print("존재하지 않는 type")
#         quit()
#
#     # 주의!! sequential이 아닌 방식의 경우, data_y가 [a,b]형태가 되어야함
#     return data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v
#
# def tttttt(csv_file_name, is_one_hot_encoding,type_train):
#     data_x_p_black = []  # 흑 정책망 input
#     data_x_p_white = []  # 백 정책망 input
#     labels_p_black = []  # 흑 정책망 레이블
#     labels_p_white = []  # 백 정책망 레이블
#     data_x_v = []  # 가치망 input
#     labels_v = []  # 가치망 레이블
#     data_y_p_black = None
#     data_y_p_white = None
#     data_y_v = None
#
#     if type_train >= 3:
#         print(f"존재 하지 않는 type_train : {type_train}")
#         quit()
#
#     print("\n데이터 셋 로딩 시작..")
#     with open(csv_file_name, 'r') as f:
#         next(f, None)
#         reader = csv.reader(f)
#         count_read = 0
#         skip_count = 0
#         # 헤더 : move 위치 / black_value / white_value / 상태~
#         for row in reader: # row는 문자열 리스트
#             count_read += 1
#             if float(row[1]) <= -100 or float(row[2]) <= -100: # 승부 판별 불가능
#                 skip_count+=1
#                 continue
#             if int(float(row[1]) == 1) and int(float(row[2]) == 0): # 흑이 이기는 경우
#                 if type_train != 0: continue
#                 labels_p_black.append(int(float(row[0])))
#                 data_x_p_black.append(row[3:])
#             elif int(float(row[1]) == 0) and int(float(row[2]) == 1): # 백이 이기는 경우
#                 if type_train != 1: continue
#                 labels_p_white.append(int(float(row[0])))
#                 data_x_p_white.append(row[3:])
#             else:
#                 # 무승부는 따로 학습 X
#                 if not (float(row[1]) == 0.5 and float(row[2]) == 0.5): # 무승부도 아닌 경우
#                     print(f"잘못된 value : 행 : {count_read-1} / 흑 : {row[1]} , 백 : {row[2]}")
#                     skip_count += 1
#                     continue  # 일단 스킵
#
#             # 가치망 데이터는 흑이 이길 확률
#             if type_train == 2:
#                 data_x_v.append(row[3:])
#                 labels_v.append(row[1])
#             if count_read % 4000 == 0:
#                 print("현재까지 읽은 row 수 :",count_read)
#
#     if len(data_x_p_black) >=1:
#         data_x_p_black = np.array(data_x_p_black, dtype=np.float32)
#     if len(data_x_p_white) >= 1:
#         data_x_p_white = np.array(data_x_p_white, dtype=np.float32)
#     if len(labels_p_black) >= 1:
#         labels_p_black = np.array(labels_p_black, dtype=np.int32)
#         data_y_p_black = labels_p_black
#         data_y_p_black = data_y_p_black.astype(dtype=np.float32)
#     if len(labels_p_white) >= 1:
#         labels_p_white = np.array(labels_p_white, dtype=np.int32)
#         data_y_p_white = labels_p_white
#         data_y_p_white = data_y_p_white.astype(dtype=np.float32)
#     if len(labels_v) >= 1:
#         labels_v = np.array(labels_v, dtype=np.float32)
#         data_y_v = labels_v
#         data_y_v = data_y_v.astype(dtype=np.float32)
#
#
#     if is_one_hot_encoding:
#         print("0 1만으로 표현하지 않으므로 사용 X")
#         quit()
#         # a = np.array(labels_p_black)
#         # b = np.zeros((len(labels_p_black), 225))
#         # b[np.arange(len(labels_p_black)), a] = 1
#         # data_y_p_black = b
#         #
#         # a = np.array(labels_p_white)
#         # b = np.zeros((len(labels_p_white), 225))
#         # b[np.arange(len(labels_p_white)), a] = 1
#         # data_y_p_white = b
#         #
#         # a = np.array(labels_v)
#         # b = np.zeros((len(labels_v), 225))
#         # b[np.arange(len(labels_v)), a] = 1
#         # data_y_v = b
#
#     return data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v
#
#
# if __name__ == '__main__':
#     get_dataset('convert_data/csv_data/short.csv',False,'seperate',0)