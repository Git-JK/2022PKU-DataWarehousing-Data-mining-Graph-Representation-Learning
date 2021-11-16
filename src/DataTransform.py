import csv
import os

def txt2csv(file_path, dst_path, head_line=''):
    f = open(file_path, 'r')
    data = []
    data.append(head_line.split(','))
    for line in f.readlines():
        line = line.strip('\n')
        data.append(line.split(' '))
    f.close()
    f = open(dst_path, 'w', encoding='utf-8', newline="")
    writer = csv.writer(f)
    for line in data:
        writer.writerow(line)
    f.close()

txt2csv('./dataset/cora/edge_list.txt', './dataset/cora/edge_list.csv' , "src,dst")
txt2csv('./dataset/chameleon/edge_list.txt', './dataset/chameleon/edge_list.csv' , "src,dst")
txt2csv('./dataset/actor/edge_list.txt', './dataset/actor/edge_list.csv' , "src,dst")