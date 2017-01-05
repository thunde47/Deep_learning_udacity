
from PIL import Image
from resizeimage import resizeimage

width=128
height=64
for i in range(33401):
	image_name=str(i+1)+'.png'
	with open('train/'+image_name, 'r+b') as f:
		with Image.open(f) as image:
			cover=resizeimage.resize_cover(image,[width,height],validate=False)
			cover.save('train_mod_'+str(width)+'x'+str(height)+'/'+image_name,image.format)



	
