import zipfile
import shutil
import os
from pathlib import Path


ori_data_dir = Path('./data/nyuv2.zip')
my_data_dir = Path('./sync')
extracted_dir = Path('./nyuv2/extracted')

# 1.load data
print('Loading from original dataset...')

if not my_data_dir.exists():
    print('Unzipping dataset...')
    with zipfile.ZipFile(ori_data_dir) as zf:
        zf.extractall()
    print('Unzip finished!')

if not extracted_dir.exists():
    os.makedirs(extracted_dir)

count_total = 0
index_my = 0
for curDir, dirs, files in os.walk(my_data_dir):
    for numer_in_curDir, file in enumerate(files):
        if file.startswith('rgb'):
            count_total += 1
            print(curDir)
            print(file)
            # print(file[-7], file[-6], file[-5])
            print('nummer in curDir is:%d' % numer_in_curDir)
            print('count_total=%d' % count_total)
            if count_total % 3 == 0:
                move_path_rgb = os.path.join('./', curDir, file)
                file_dpt = 'sync_depth_'+'00' + \
                           str(file[-7]) + str(file[-6]) + str(file[-5]) + '.png'
                move_path_dpt = os.path.join('./', curDir, file_dpt)
                my_file = os.path.join(extracted_dir,
                                       'rgb_'+'0'*(5-len(str(index_my)))+str(index_my)+'.jpg')
                my_file_dpt = os.path.join(extracted_dir,
                                           'dpt_'+'0'*(5-len(str(index_my)))+str(index_my)+'.png')
                shutil.move(move_path_rgb, my_file)
                shutil.move(move_path_dpt, my_file_dpt)
                index_my += 1
                print('file copied:' + move_path_rgb)
        # else:
        #     if str(file[-7])+str(file[-6])+str(file[-5]) == '0'*(3-len(str(index_my)))+str(index_my):
        #         print('Check Ok!')


print('Tatal num of data pairs is:%d' % count_total)
print('Extracted num of data paris is:%d' % index_my)
shutil.rmtree('sync')

from PIL import Image
import paddle.vision.transforms as transforms

dpt_test = Image.open('./nyuv2/extracted/dpt_00000.png')
print(dpt_test)
tr = transforms.ToTensor()
dpt_test_tr = tr(dpt_test)
print(dpt_test_tr)
print(dpt_test_tr.shape)
