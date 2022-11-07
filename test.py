
from collections import defaultdict, deque

dq = deque(maxlen=5)
for i in range (40):
    dq.extend(str(i))
    print()