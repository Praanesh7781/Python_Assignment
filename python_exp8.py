# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 22:30:32 2023

@author: Lenovo
"""

color = {}
n = int (input("Enter the number of elements:"))
for i in range(n):
    key = input("Enter the key: ")
    value = input("Enter the value: ")
    color[key]=value
print("Dictionary elements are:\n")
for key,value in color.items():
    print(key,value)
