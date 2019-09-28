times = input('Please enter two times formatting "HH:MM": ')
h1 = int(times[0]) * 10 + int(times[1])
m1 = int(times[3]) * 10 + int(times[4])
h2 = int(times[6]) * 10 + int(times[7])
m2 = int(times[9]) * 10 + int(times[10])
#print(h1, m1, h2, m2)

s1 = h1 * 3600 + m1 * 60
s2 = h2 * 3600 + m2 * 60
print("The difference between the two times is " + str(abs(s2 - s1)) + " seconds.")