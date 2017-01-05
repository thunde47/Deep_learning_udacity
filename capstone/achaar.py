import numpy as np
from PIL import Image
import scipy.io as sio
from six.moves import cPickle as pickle
from scipy.ndimage.filters import uniform_filter

total_images=33401
images_used=8000
width=128
height=64
image_path='train_mod_'+str(width)+'x'+str(height)+'/'


struct=sio.loadmat('train/digitStruct_v7.mat')
lengths=np.zeros(images_used)
digits=np.full((images_used,5),10.)
for i in range(images_used):
	
	_,length=struct['digitStruct']['bbox'][0][i].shape
	if length>5:
		length=5
	lengths[i]=length
	
	for j in range(length):
		digit=struct['digitStruct']['bbox'][0][i]['label'][0][j][0][0]
		if digit>9:
			digits[i][j]=0.
		else:
			digits[i][j]=digit


digit1=np.empty(images_used)
digit2=np.empty(images_used)
digit3=np.empty(images_used)
digit4=np.empty(images_used)
digit5=np.empty(images_used)

for i in range(images_used):
	digit1[i]=digits[i][0]
	digit2[i]=digits[i][1]
	digit3[i]=digits[i][2]
	digit4[i]=digits[i][3]
	digit5[i]=digits[i][4]

target=[lengths,digit1,digit2,digit3,digit4,digit5]
dataset=[]

for i in range(images_used):
	image_name=str(i+1)+'.png'
	image=Image.open(image_path+image_name)
	dataset.append(np.array(image))

def window_stdev(arr,radius):
	c1=uniform_filter(arr,radius*2,mode='constant',origin=-radius)
	c2=uniform_filter(arr*arr,radius*2,mode='constant',origin=-radius)
	return ((c2-c1*c1)**.5)[:-radius*2+1,:-radius*2+1]

mean=np.mean(dataset)
stddev=np.std(dataset)


#stddev=window_stdev((np.asarray(dataset)).reshape(images_used*width*height*3),20)	
#print((np.asarray(dataset)).reshape(images_used*width*height*3))
print('standard deviation=',stddev)

dataset=.5*(dataset-mean)/stddev

training_samples=7000
train_target=[]
test_target=[]
train_dataset=dataset[0:training_samples]
test_dataset=dataset[training_samples:total_images]
for i in range(6):
	train_target.append(target[i][0:training_samples])
	test_target.append(target[i][training_samples:total_images])
print(train_dataset.shape)
print(test_dataset.shape)

achaar_file = 'svhn.pickle'
try:
  f = open(achaar_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_target': train_target,
	'test_dataset': test_dataset,
    'test_target': test_target,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', achaar_file, ':', e)
  raise
