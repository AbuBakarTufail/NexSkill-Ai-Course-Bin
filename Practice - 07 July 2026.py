import numpy as np
rows = int(input("Enter the number of rows to read: "))

for counter in range(1, rows + 1):
    for innerCounter in range(1, counter + 1):
        print("*", end="")
    print()