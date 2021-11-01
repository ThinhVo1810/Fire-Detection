# Fire-Detection

## Prepare data

Tải về mạng vgg16

```
cd data
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

 Ở file [prepare_data.py](https://github.com/ThinhVo1810/Fire-Detection/blob/main/prepare_data.py) chúng ta cần sửa lại url thành [link này](https://l.facebook.com/l.php?u=https%3A%2F%2Fdrive.google.com%2Fu%2F1%2Fuc%3Fid%3D1G1cRVsl_F46ea19eRH6A-ZU_h5aq5nFI%26fbclid%3DIwAR2yRRYx6DyjI0Vk3mECa4te6b1XNvW2lj7R6VxfqQCjPVJYo0wB67-fiE4&h=AT1rhqmartnHz_qyBXENq6cRydBsj01roiuvCeli3am62aP2luH8VYhdPgZhe50YWtMwVn2rJAkMjlQAom4kgSNx1GYSeBf2PaBjbx2vujdS_sJKk9LmhPHeKILSdA)
 sau đó ở dòng [16](https://github.com/ThinhVo1810/Fire-Detection/blob/13bd739f4b15271b64113ad890445b10617dc907/prepare_data.py#L11) sửa lại thành *VOC.tar.xz*

## Training 
```
python train.py
```

