import csv

# 前提是原txt文件用,隔开
out = open('output.csv', 'w', newline='')  # 要转成的.csv文件，先创建一个.csv文件
csv_writer = csv.writer(out, dialect='excel')

f = open("output.txt", "r")
for line in f.readlines():
    line = line.replace(',', '\t')  # 将每行的逗号替换成空格
    L = line.split()  # 将字符串转为列表，从而可以按单元格写入csv
    csv_writer.writerow(L)
