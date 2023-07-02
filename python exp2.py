Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> str1=str(input("Enter the first string\n"))
... str2=str(input("Enter the second string\n"))
... str3=''
... temp=0
... len1=len(str1)
... len2=len(str2)
... for i in range(0,len2):
...     for i in range(temp,len1):
...         if(str2[i]==str1[i]):
...         str3=str3+str1[]
...         temp=i+1
...         break
... if(str2==str3):
...     print("yes")
... else:
