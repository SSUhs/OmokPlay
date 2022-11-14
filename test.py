import re
import numpy as np


child_priors = np.zeros([362], dtype=np.float32)
child_total_value = np.zeros(
    [362], dtype=np.float32)
child_number_visits = np.zeros(
    [362], dtype=np.float32)

def child_U(child_priors,child_total_value,child_number_visits):
    return math.sqrt(number_visits) * (child_priors / (1 + child_number_visits))

child_U(child_priors,child_total_value,child_number_visits)