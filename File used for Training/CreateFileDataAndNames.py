import os
import argparse

# cách chạy file với cmd
# python CreateFileDataAndNames.py -d  '/Users/trunghieu/Desktop/HCMUS/HocThongKe/YOLO-3/darknet'


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--darknet', required=True,
                help='absolute path to darknet folder with ending in "/" ')
args = ap.parse_args()

path_to_darket = args.darknet

counter = 0

os.chdir(path_to_darket)
with open('classes.names', 'w') as names, \
    open('data/images/classes.txt', 'r') as txt:
     for line in txt:
         names.write(line)
         counter += 1

with open('custom_data.data', 'w') as data:
    data.write('classes = ' + str(counter) + '\n')
    data.write('train = train.txt' + '\n')
    data.write('valid = val.txt' + '\n')
    data.write('names = classes.names' + '\n')
    data.write('backup = backup')

