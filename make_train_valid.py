#!/usr/bin/env python3
import glob
import os
import random

##init seed
random.seed(114514)

print("make train.txt  test.txt script")
print("==================================\n\n")
path = input("if make files in {} write Y else write fullpath  ".format(os.getcwd()))
if "Y" == path or "y" == path or "" == path:
    path = os.getcwd()
exh = input("fileformat? jpg? bmp? >>")

dirctory = input("write images dirctory full-path !! this script find bmp recurrentry::")
# print("dirctory::",dirctory)
assert dirctory != "", "please write fullpath"
train_rate = float(input("train data rate ex:) train data rate=0.7 test data rate=0.3 => 0.7 "))
print("train_rate=", train_rate)
if dirctory[-1] != "/": dirctory += "/"
dirctory += "**/*" + exh
print(dirctory)
imgList = glob.glob(dirctory, recursive=True)
print("find", len(imgList), "files")

IMG_NUM = len(imgList)
trainNUM = int(IMG_NUM * train_rate)
testNUM = IMG_NUM - trainNUM
trainIMG_LIST = random.sample(imgList, trainNUM)
testIMG_LIST = list(set(trainIMG_LIST) ^ set(imgList))

# assert len(testIMG_LIST)!=0, "not enough data. check dirctory!"
print("train data => ", trainNUM)
print("test data  => ", testNUM)

train_file = open(os.path.join(path, "train.txt"), "w")
test_file = open(os.path.join(path, "test.txt"), "w")

train_file.write("\n".join(trainIMG_LIST))
test_file.write("\n".join(testIMG_LIST))
print("success")
