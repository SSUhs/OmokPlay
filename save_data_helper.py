import pickle

# lr_mulitplier과 deque를 저장
class save_data_helper():
    def __init__(self,train_num,board_width,learn_rate,lr_multiplier,data_buffer):
        self.is_loaded = True
        self.train_num = train_num
        self.board_width = board_width
        self.learn_rate = learn_rate
        self.lr_multiplier = lr_multiplier
        self.data_buffer = data_buffer

    def save_model_data(self):
        if not self.is_loaded:
            print("learning_rate_multiplier 저장에 실패하였습니다")
            return
        pickle.dump(self, open(f'/content/drive/MyDrive/tf_train_{self.board_width}_{self.train_num}.pickle', 'wb'),
                    protocol=2)
