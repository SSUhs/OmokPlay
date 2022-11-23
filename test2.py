import copy
import numpy as np

class t:
    def get_best_idx(self, board):
        probs = self.model.predict(input)
        probs_tmp = copy.deepcopy(probs)
        while True:
            best_index = np.argmax(probs_tmp[0])
            # 이미 돌이 있는 자리를 선택하거나 금수에 놓은 경우
            if self.is_banned_pos(board, best_index):
                probs_tmp[0][best_index] = -1  # 금수 자리는 선택 불가능 하게 설정
                continue
            else:
                break
        return best_index

