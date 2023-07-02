# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:28:30 2023

@author: Lenovo
"""

setx = set(["apple", "mango",45,33,99])
sety = set(["mango", "orange",33])
setz = set(["mango"])
print("x: ",setx)
print("y: ",sety)
print("z: ",setz,"\n")
print("Checking whether X is subset of Y...... \n")
if(setx.issubset(sety)==True):
    print("The set X is the subset of set Y \n")
else:
    print("The set X is not the subset of set Y \n")
print("Checking whether Y is subset of X...... \n")
if(sety.issubset(setx)==True):
    print("The set Y is the subset of set X\n")
else:
    print("The set Y is not the subset of set X\n")
print("Checking whether Y is subset of Z...... \n")
if(sety.issubset(setz)==True):
    print("The set Y is the subset of set Z\n")
else:
    print("The set Y is not the subset of set Z\n")
print("Checking whether Z is subset of Y...... \n")
if(setz.issubset(sety)==True):
    print("The set Z is the subset of set Y\n")
else:
    print("The set Z is not the subset of set Y\n")