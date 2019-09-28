n = int(input("Please input the number of lines of the triangle: "))
for i in range(1, n+1):
    padding = (n - i) * " "
    print("%s%s" % (padding, "*" * (2 * i - 1)))