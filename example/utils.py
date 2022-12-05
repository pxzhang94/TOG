import os

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def process_label(olabel):
    if olabel == 0:
        nlabel = 4
    if olabel == 1:
        nlabel = 1
    if olabel == 2:
        nlabel = 14
    if olabel == 3:
        nlabel = 8
    if olabel == 4:
        nlabel = 39
    if olabel == 5:
        nlabel = 5
    if olabel == 6:
        nlabel = 2
    if olabel == 7:
        nlabel = 15
    if olabel == 8:
        nlabel = 56
    if olabel == 9:
        nlabel = 19
    if olabel == 10:
        nlabel = 60
    if olabel == 11:
        nlabel = 16
    if olabel == 12:
        nlabel = 17
    if olabel == 13:
        nlabel = 3
    if olabel == 14:
        nlabel = 0
    if olabel == 15:
        nlabel = 58
    if olabel == 16:
        nlabel = 18
    if olabel == 17:
        nlabel = 57
    if olabel == 18:
        nlabel = 6
    if olabel == 19:
        nlabel = 62
    return nlabel