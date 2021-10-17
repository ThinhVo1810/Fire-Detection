import tarfile
import gdown
import os

# Tai xuong du lieu
data_dir = './data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

url = 'https://drive.google.com/u/0/uc?id=1ydVpGphAJzVPCkUTcJxJhsnp_baGrZa7'
target_path = os.path.join(data_dir, 'fire_detection.tar')

if not os.path.exists(target_path):
    gdown.download(url, target_path, quiet=False)

    tf = tarfile.open(target_path)
    tf.extractall(data_dir)