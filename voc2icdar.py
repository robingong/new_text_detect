import os
import argparse
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser()
parser.add_argument("--orgin_data_dir", required=False, type=str, default='/data/文本检测/custom_data_thai/corpus_thai/img/print')
parser.add_argument("--target_data_dir", required=False, type=str, default='/data/文本检测/custom_data_thai/icdar')
flags = parser.parse_args()

for parent, dirnames, filenames in os.walk(os.path.join(flags.orgin_data_dir)):
    for filename in filenames:
        if filename.endswith('xml'):
            icdar_text = ''
            with open(os.path.join(parent, filename), 'r', encoding='utf-8') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                for object in root.findall('object'):
                    name = object.find('name').text  # 子节点下节点name的值
                    #print(name)
                    bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
                    xmin = bndbox.find('xmin').text
                    ymin = bndbox.find('ymin').text
                    xmax = bndbox.find('xmax').text
                    ymax = bndbox.find('ymax').text
                    #print(xmin, ymin, xmax, ymax)
                    # ICDAR 左上、右上、右下、左下, 语言, 文字 (xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin, ina, ###
                    icdar_text = icdar_text + xmin + ',' + ymax + ',' + xmax + ',' + ymax + ',' + xmax + ',' + ymin + ',' + xmin + ',' + ymin + ',' + 'ina,text\n'

            filename_seq = filename.split('.')
            if len(filename_seq) ==2:
                txt_filename = 'thai_'+filename_seq[0]+'.txt'
                img_filename = filename_seq[0]+'.jpg'
                img_new_filename = 'thai_'+filename_seq[0]+'.jpg'
            else:
                print("================================", filename)
                txt_filename = 'thai_'+filename_seq[0]+'.'+filename_seq[1]+'.txt'
                img_filename = filename_seq[0] + '.' + filename_seq[1] + '.jpg'
                img_new_filename = 'thai_' + filename_seq[0] + '.' +filename_seq[1]+ '.jpg'
            with open(os.path.join(flags.target_data_dir, 'txt', txt_filename), 'w', encoding='utf-8') as f:
                f.write(icdar_text)
        #else:
            os.system('cp ' + os.path.join(parent, img_filename) + ' ' + os.path.join(flags.target_data_dir, 'image', img_new_filename))
            #os.system('mv ' + os.path.join(flags.target_data_dir, 'image', filename) + ' ' + os.path.join(flags.target_data_dir, 'image', img_filename))
