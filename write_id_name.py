import os

id_img = []
directory = '/home/huynth/ImageProcessing/data/VOC2020/JPEGImages/'
test_file = open('val.txt', 'w')
for dir_name, _, filenames in os.walk(directory):
    for filename in filenames:
        id_img.append(os.path.splitext(filename)[0])
    
for elements in id_img:
    test_file.write(elements + "\n")
test_file.close()
