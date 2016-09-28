#coding:utf8
from tomorrow import threads
from time import sleep
from tqdm import *

@threads(5)
def a(num):
    sleep(1)
    print num

'''
if __name__ == '__main__':

    for each in tqdm(range(100)):
        a(each)

'''
