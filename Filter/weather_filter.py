from numpy import imag
import imgaug.augmenters as iaa
import imageio
import imgaug as ia


path = 'data/VOC2020/JPEGImages/wildfire.jpg'
image = imageio.imread(path)
#ia.imshow(image)

# Create Fast Snowy
aug = iaa.FastSnowyLandscape(
    lightness_threshold=140,
    lightness_multiplier=2.5
)

img_snow = aug(image=image)
#ia.imshow(img_snow)
imageio.imwrite('result/img_fastsnow.jpg', img_snow)

# Create Cloud
aug = iaa.Clouds()
img_cloud = aug(image=image)
#ia.imshow(img_cloud)
imageio.imwrite('result/img_cloud.jpg', img_cloud)

# Create Fog filter
aug = iaa.Fog()

img_fog = aug(image=image)
#ia.imshow(img_fog)
imageio.imwrite('result/img_fog.jpg', img_cloud)

# Create snow
aug = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))

img_snow_flake = aug(image=image)
#ia.imshow(img_snow_flake)
imageio.imwrite('result/img_snow.jpg', img_cloud)


aug = iaa.imgcorruptlike.Snow(severity=2)
img_Snow = aug(image=image)
#ia.imshow(img_Snow)
imageio.imwrite('result/Snow.jpg', img_Snow)

# Create DefocusBlur
aug = iaa.imgcorruptlike.DefocusBlur(severity=4)
img_defocus = aug(image=image)
#ia.imshow(img_defocus)
imageio.imwrite('result/defocus.jpg', img_defocus)

# Create MotionBlur
aug = iaa.imgcorruptlike.MotionBlur(severity=4)
img_motionblur = aug(image=image)
#ia.imshow(img_motionblur)
imageio.imwrite('result/motionBlur.jpg', img_motionblur)

# Create Spatter
aug = iaa.imgcorruptlike.Spatter(severity=2)
img_spatter = aug(image=image)
#ia.imshow(img_spatter)
imageio.imwrite('result/spatter.jpg', img_spatter)

# anh co che =))
aug = iaa.imgcorruptlike.Pixelate(severity=5)
img_pix = aug(image=image)
#ia.imshow(img_pix)
imageio.imwrite('result/pix.jpg', img_pix)

aug = iaa.imgcorruptlike.ElasticTransform(severity=5)
img_elastic = aug(image=image)
ia.imshow(img_elastic)
imageio.imwrite('result/img_elastic.jpg', img_elastic)