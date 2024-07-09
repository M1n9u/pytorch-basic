import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

import os
import glob
from io import open

filenames = glob.glob('data/names/*.txt')
country = []
names = {}
for files in filenames:
    country_name = os.path.splitext(os.path.basename(files))[0]
    country.append(country_name)
    lines = open(files, encoding='utf-8').read().strip().split('\n')
    lines = [unicodeToAscii(line) for line in lines]
    names[country_name] = lines

print(country)