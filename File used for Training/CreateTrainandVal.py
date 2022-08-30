import os
import argparse


# cách chạy file với cmd
# python CreateTrainandVal.py -i '/Users/trunghieu/Desktop/HCMUS/HocThongKe/YOLO-3/darknet/data/images' \
#                             -d  '/Users/trunghieu/Desktop/HCMUS/HocThongKe/YOLO-3/darknet'


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True,
                help='absolute path to images folder with ending in "/" ')

ap.add_argument('-d', '--darknet', required=True,
                help='absolute path to darknet folder with ending in "/" ')


args = ap.parse_args()
# đường dẫn đến images - argument
path_to_images = args.images

path_to_darket = args.darknet

os.chdir(path_to_images)

p = []

for _,_,files in os.walk('.'):
    # duyệt qua toàn bộ file
    for f in files:
        if f.endswith('.jpg') or f.endswith('.jpeg'):
            path_to_save_txt_files = 'data/images/' + f

            p.append(path_to_save_txt_files + '\n')

# Chia tập train và tập test 80-20
p_test = p[:int(len(p) * 0.2)]
p = p[int(len(p) * 0.2):]

# lưu file train.txt và test.txt
os.chdir(path_to_darket)

with open('train.txt','w') as f:
    for e in p:
        f.write(e)
with open('val.txt', 'w') as f:
    for e in p_test:
        f.write(e)


