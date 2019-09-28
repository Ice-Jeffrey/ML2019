PI = 0
for i in range(1, 100000, 4):
    PI += 4 / i - 4 / (i+2)
print("The estimated value of PI after 100000 iterations is " + str(PI))

radius = [10, 100, 250]
for r in radius:
    area = round(r * r * PI, 3)
    print("The area of the circle with the radius " + str(r) + " is " + str(area))