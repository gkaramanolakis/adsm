import os
import sys
import pickle
from collections import Counter
import shutil

def check_existence(fpath):
    if not os.path.exists(fpath):
        print("[ERROR] File nonexistent: {}.format()")
        sys.exit(1)


def mkdir(fpath, delete_old=True):
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    elif delete_old:
        shutil.rmtree(fpath)
        os.makedirs(fpath)


def save_dict(python_dict, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(python_dict, f)
    return


def cutoff_taglist(filename,cutoff=False):
    with open(filename) as f:
        #tags    = [line.split('\t')[1].split() for line in f.readlines()]
        tags = list([word for line in f for word in line.split('\t')[1].strip('\n').split(' ') ]) # gia na doulepsei allaksa to sfx_plain.py
    #totaltags   = len([t for l in tags for t in l])
    c           = Counter()
    for l in tags:
        c.update([l])
    if cutoff:
        for k in c.keys():
            if k.isdigit() or c[k] <= 5:
                c.pop(k)
    return [k for k in c.keys()],c


def soundid_words(tagfile):
    #return a dictionary sound_id : tags
    with open(tagfile, 'r') as f:
        lines = f.readlines()
        id_list = [line.split('\t')[0] for line in lines]
        tagging_list = [line.split('\t')[1].strip('\n').split(' ') for line in lines]
    return {id_list[i]: tagging_list[i] for i in range(len(id_list))}


def word_soundid(wordlist, soundidwords):
    #return a dictionary tag : sound_id's
    return {wordlist[i]: [sound for sound in soundidwords if wordlist[i] in soundidwords[sound]] for i in range(len(wordlist))}


def save_dict_as_txt(savepath,dictionary):
    with open(savepath,'w') as f:
        for key,value in dictionary.items():
            f.write(key.encode('utf8'))
            for v in value:
                f.write(' %f'%v)
            f.write('\n')
    return

