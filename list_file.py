from os import listdir
from os.path import isfile, join
from random import shuffle
from math import floor
import csv

mypath = "/media/alissonsales/Files/base_dados/pt_spec/"
pt_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = "/media/alissonsales/Files/base_dados/es_spec/"
es_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

mypath = "/media/alissonsales/Files/base_dados/en_spec/"
en_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

shuffle(pt_files)
shuffle(es_files)
shuffle(en_files)

#tamanho do conjunto de teste em %
test_size = 0.1

pt_test_size = floor(len(pt_files)*test_size)
es_test_size = floor(len(pt_files)*test_size)
en_test_size = floor(len(pt_files)*test_size)

pt_test_files, pt_train_files = pt_files[:pt_test_size], \
                                pt_files[pt_test_size:]

es_test_files, es_train_files = es_files[:es_test_size], \
                                es_files[es_test_size:]

en_test_files, en_train_files = en_files[:en_test_size], \
                                en_files[en_test_size:]

test_files =  pt_test_files + es_test_files + en_test_files
train_files = pt_train_files + es_train_files + en_train_files

shuffle(test_files)
shuffle(train_files)

path = "/media/alissonsales/Files/base_dados/"

with open(path + 'train.csv', 'w') as fw:
    for e in train_files:
        file_name = e.split(".")[0]
        lang = file_name.split("_")[1]
        if lang == 'pt':
            lang_i = 0
        elif lang == 'es':
            lang_i = 1
        elif lang == 'en':
            lang_i = 2

        writer = csv.writer(fw, delimiter=',')
        writer.writerow([file_name, lang_i])
fw.close()

with open(path + 'test.csv', 'w') as fw:
    for e in test_files:
        file_name = e.split(".")[0]
        lang = file_name.split("_")[1]
        if lang == 'pt':
            lang_i = 0
        elif lang == 'es':
            lang_i = 1
        elif lang == 'en':
            lang_i = 2

        writer = csv.writer(fw, delimiter=',')
        writer.writerow([file_name, lang_i])
fw.close()
