#coding=utf-8

from urllib import urlretrieve
import fileinput

for line in fileinput.input("D:\\data.txt"):
    urlretrieve('http://img.wkzf.com/' + line.strip() + '.DL', 'd:/img/' + line.strip() + '.jpg')