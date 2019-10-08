from math import *

Letter_American_height = 11.0
Letter_American_width = 8.5
inch_to_millimeters = 25.4
Letter_height = round(Letter_American_height * inch_to_millimeters, 1)
Letter_width = round(Letter_American_width * inch_to_millimeters, 1)
print("The size of a letter in American standard is " + str(Letter_American_height) + " * " + str(Letter_American_width))
print("The size of a letter in millimeters is " + str(Letter_height) + " * " + str(Letter_width))
print("The size of an A4 letter in millimeters is 297.0 * 210.0")

delta_height = round(abs(297.0 - Letter_height), 1)
delta_width = round(abs(210.0 - Letter_width), 1)
print("The difference in width and height between A4 letter and US letter is " + str(delta_height) + " in height and "
       + str(delta_width) + " in width.")