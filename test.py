import re

s = ' tensorflow   13  5000'

l = re.split(r'[ ]+', s)
print(l)