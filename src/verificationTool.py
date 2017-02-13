#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lugao
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def build(instructions, w, h):
    np.random.seed(w*h)
    v = np.zeros((h,w), dtype = np.int32)
    for l in instructions:
        lsplit = l.split(',')
        x = int(lsplit[1])
        y = int(lsplit[2])
        if(lsplit[0] == 'FILL'):
            s = int(lsplit[3])
            v[y:y+s,x:x+s] = np.random.randint(1, 100)
        else:
            v[y:y+1,x:x+1] = 0
    return v
def printImg(v, colorMap = "Greys", size = 10):
    plt.rcParams["figure.figsize"] = (size,size)
    plt.figure(1)
    plt.imshow(v, interpolation="nearest",vmin=0.5, cmap=colorMap)
    plt.show()
def analyseSolution(infile = 'input_e.txt', outfile = 'output_e.txt', size = 5):
    with open(infile, 'r') as fin:
        [w,h] = [int(num) for num in fin.readline().split(',')]
        a = np.array([[1 if c == '#' else 0 for c in list(line.split()[0])] for line in fin])
        with open(outfile, 'r') as fout:
            ops = [line for line in fout]
            b = build(ops, w, h)
            
            cmap = cm.get_cmap('rainbow')
            cmap.set_under('white')
            printImg(b, cmap)
            for y in range(h):
                for x in range(w):
                    if(a[y,x] == 1 and b[y,x] == 0):
                        print("not filled ({},{})".format(x,y))
                    if(a[y,x] == 0 and b[y,x] == 1):
                        print("not empty  ({},{})".format(x,y))
            bugs = (abs(a - (b != 0)) != 0).sum()  
            if(bugs == 0):
                print("Valid solution!")
            else:
                print("{} bugs found!".format(bugs))

analyseSolution('../input_0.txt', '../output_0.txt',10)