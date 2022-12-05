CONST_SUCCESS = 0  # 오류 X
CONST_UNKNOWN = -100  # 알 수 없는 오류
CONST_WRONG_POSITION = -101  # 위치 아예 벗어나서 잘못 놓았을 경우
CONST_BANNED_POSITION = -102  # 금수 자리에 놓은 경우
CONST_GAME_FINISH = -103

BANNED_OK = -1000  # 문제 없는 배치
BANNED_33 = -1001  # 3x3으로 배치한 경우
BANNED_44 = -1002  # 4x4으로 배치
BANNED_6 = -1003  # 6목