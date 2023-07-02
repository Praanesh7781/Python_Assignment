# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:10:47 2023

@author: Lenovo
"""

def str1(t):
    res = ' '
    for item in t:
        res = res + item
        return res
num=int(input("Enter number of elements in tuple :"))
l=list()
for i in range(0,num):
    e=input("enter the value :")
    l.append(e)
a=tuple(l)
print("The string is", str1(a))