import string
import random

ID_LEN = 10

def rand_id():
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(ID_LEN))