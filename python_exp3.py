# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 21:59:24 2023

@author: Lenovo
"""

num=int(input("Enter number of elements in list :"))
l=list()
for i in range(0,num):
    e=input("enter the value :")
    l.append(e)
print ("Original list : " + str(l))
k=input("Enter the element to be searched in the list : ")
res=len(l)-l.index(k)
print("The required Negative index : -",str(res))
print("The required Positive index : ",l.index(k))
'''res = ~l[::-1].index(k)
print("The required Negative index : ",str(res))'''