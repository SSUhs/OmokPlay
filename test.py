# import numpy as np
# import csv
# import keras.backend as K
#
# def convert_load_dataset(csv_file_name, is_one_hot_encoding,type_train):
#     board_size = 15
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
#
#     print("\n데이터 셋 로딩 시작..")
#     with open(csv_file_name, 'r') as f:
#         next(f, None)
#         reader = csv.reader(f)
#         count_read = 0
#         skip_count = 0
#         # 헤더 : move 위치 / black_value / 누가 돌을 놓을차례(흑1, 백2) / 상태~
#         for row in reader: # row는 문자열 리스트
#             count_read += 1
#             # if float(row[1]) <= -100: # 승부 판별 불가능
#             #     skip_count+=1
#             #     continue
#             if type_train == 0: # 흑,백 정책망 학습
#                 # if float(row[1]) <= 0.5 and int(float(row[2]) == 1):  # 흑이 이기거나 비기는 경우면서 흑이 돌을 놓을 차례인 경우
#                 label = int(float(row[0])) # 정답 라벨
#                 board_2nd = convert_1nd_board_to_2nd(np.array(row[3:]),board_size=board_size) # 2차원 형태로 state 변경
#                 rotate_12dir_states = get_rotate_board_12dir(board_2nd) # 12번 뒤집은 state
#                 rotate_12dir_labels = get_rotate_label(label,board_size)
#                 for i in range(len(rotate_12dir_states)):
#                     data_x_p_black.append(list(convert_2nd_board_to_1nd(rotate_12dir_states[i])))
#                     labels_p_black.append(int(rotate_12dir_labels[i]))
#             elif type_train == 1: # 백 정책망 학습
#                 # if float(row[1]) >= 0.5 and int(float(row[2]) == 2):  # 백이 이기거나 비기는 경우면서 백이 돌을 놓을 차례인 경우
#                 # 일단 전부 다 학습
#                 label = int(float(row[0])) # 정답 라벨
#                 board_2nd = convert_1nd_board_to_2nd(np.array(row[3:]),board_size=board_size) # 2차원 형태로 state 변경
#                 rotate_12dir_states = get_rotate_board_12dir(board_2nd) # 12번 뒤집은 state
#                 rotate_12dir_labels = get_rotate_label(label,board_size)
#                 for i in range(len(rotate_12dir_states)):
#                     data_x_p_white.append(convert_2nd_board_to_1nd(list(rotate_12dir_states[i])))
#                     labels_p_white.append(int(rotate_12dir_labels[i]))
#             # elif type_train == 1: # 백 정책망 학습
#             #     if float(row[1]) >= 0.5 and int(float(row[2]) == 2):  # 백이 이기거나 비기는 경우면서 백이 돌을 놓을 차례인 경우
#                 # 일단 전부 다 학습
#                 # labels_p_white.append(int(float(row[0])))
#                 # data_x_p_white.append(row[3:])
#             elif type_train == 2:
#                 labels_v.append(float(row[1]))
#                 board_2nd = convert_1nd_board_to_2nd(np.array(row[3:]),board_size=board_size) # 2차원 형태로 state 변경
#                 rotate_12dir_states = get_rotate_board_12dir(board_2nd) # 12번 뒤집은 state
#                 for i in range(len(rotate_12dir_states)):
#                     data_x_v.append(list(convert_2nd_board_to_1nd(rotate_12dir_states[i])))
#                     labels_v.append(float(row[1]))
#             if count_read % 4000 == 0:
#                 print("현재까지 읽은 row 수 :",count_read)
#
#     if len(data_x_p_black) >= 1:
#         data_x_p_black = np.array(data_x_p_black, dtype=np.float32)
#     if len(data_x_p_white) >= 1:
#         data_x_p_white = np.array(data_x_p_white, dtype=np.float32)
#     if len(data_x_v) >= 1:
#         data_x_v = np.array(data_x_v, dtype=np.float32)
#
#     if len(labels_p_black) >= 1:
#         labels_p_black = np.array(labels_p_black, dtype=np.int32)
#         data_y_p_black = labels_p_black
#         data_y_p_black = data_y_p_black.astype(dtype=np.int32)
#     if len(labels_p_white) >= 1:
#         labels_p_white = np.array(labels_p_white, dtype=np.int32)
#         data_y_p_white = labels_p_white
#         data_y_p_white = data_y_p_white.astype(dtype=np.int32)
#     if len(labels_v) >= 1:
#         labels_v = np.array(labels_v, dtype=np.float64)
#         data_y_v = labels_v
#         data_y_v = data_y_v.astype(dtype=np.float64)
#
#     if is_one_hot_encoding:
#         print("0 1만으로 표현하지 않으므로 사용 X")
#         quit()
#
#     return data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v
#
# def reshape_to_15_15_1(data):
#     return K.reshape(data,[-1,15,15,1])
#
# # ex) 0~224 형태의 보드판을 15*15 형태의 numpy로 변경
# def convert_1nd_board_to_2nd(arr_1nd,board_size):
#     return np.reshape(arr_1nd,(board_size,board_size))
#
# def convert_2nd_board_to_1nd(arr_2nd):
#     return np.ravel(arr_2nd)
#
#
# # 0도(원본),90도,180도,270도 회전 + 각각마다 상하,좌우 반전 >> 하나의 상태로 12가지 상태 추출
# def get_rotate_board_12dir(arr):
#     all_list = []
#     for i in range(4):
#         rotates = np.rot90(arr,i)
#         flip_horizontal = np.fliplr(rotates)
#         flip_vertical = np.flipud(rotates)
#         all_list.append(rotates)
#         all_list.append(flip_horizontal)
#         all_list.append(flip_vertical)
#     return all_list
#
# def convert_to_label_start0(x, y,board_size):
#     return board_size * y + x
#
# def get_rotate_label(label,board_size):
#     arr = convert_label_to_board(label,board_size=board_size)
#     rotate_list = get_rotate_board_12dir(arr) # len = 12
#     all_list = []
#     for i in range(len(rotate_list)):
#         label_arr = rotate_list[i]
#         y,x = np.where(label_arr == 1)
#         move = convert_to_label_start0(x[0],y[0],board_size) # ndarray 형태로 나오기 때문에 0번으로 접근
#         all_list.append(move)
#     return all_list
#
#
# # move : 15*15의 경우 0~224
# # 레이블을 board형태로 전환
# def convert_label_to_board(move,board_size):
#     y = move//board_size
#     x = move-(y*board_size)
#     tmp = np.zeros([board_size,board_size])
#     tmp[y][x] = 1
#     return tmp
#
# def get_dataset(csv_name,is_one_hot_encoding,pv_type,type_train):
#     if pv_type == 'seperate':
#         data_x_p_black,data_x_p_white,data_y_p_black,data_y_p_white,data_x_v,data_y_v= convert_load_dataset(csv_name, is_one_hot_encoding=is_one_hot_encoding,type_train=type_train)
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
#
# if __name__ == '__main__':
#     get_dataset('convert_data/csv_data/short.csv',False,'seperate',0)