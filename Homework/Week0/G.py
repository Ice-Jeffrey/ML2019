print("Please input two strings: ")
str1 = input()
str2 = input()

flag = [False, False, False]
if(str1.startswith(str2)):
    flag[0] = True
if(str2 in str1):
    flag[1] = True
if(str1.endswith(str2)):
    flag[2] = True

if(flag[0]):
    print(str2 + " is the prefix of " + str1)
else:
    print(str2 + " is not the prefix of " + str1)

if(flag[1]):
    print(str2 + " is the substring of " + str1)
else:
    print(str2 + " is not the substring of " + str1)

if(flag[2]):
    print(str2 + " is the suffix of " + str1)
else:
    print(str2 + " is not the suffix of " + str1)