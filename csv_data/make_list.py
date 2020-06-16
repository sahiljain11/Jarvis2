import os

folder = "2fingers"

basedir = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(basedir, "2fingers/2fingers0.csv")

f = open(data_file, "r")
file_data = f.readlines()

first_row = file_data[0].split(",")

cols = [2, 3, 4, 5, 6, 7]

for i in range(0, len(first_row)):
    if "distal" in first_row[i]:
        cols.append(i)


print(len(cols))
print(cols)
