import numpy as np
from PIL import Image
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder

total_images=33401
images_used=10
image_path='train_mod/'
train_dataset=[]
train_labels=[]
struct=sio.loadmat('train/digitStruct_v7.mat')
#print(struct['digitStruct']['bbox'][0][998]['label'][0][1][0][0])
#print(struct['digitStruct']['bbox'][0][998])
lengths=np.zeros(images_used)
digits=np.full((images_used,5),10.)
for i in range(images_used):
	_,length=struct['digitStruct']['bbox'][0][i].shape
	lengths[i]=length
	for j in range(length):
		digit=struct['digitStruct']['bbox'][0][i]['label'][0][j][0][0]
		if digit>9:
			digits[i][j]=0.
		else:
			digits[i][j]=digit


enc=OneHotEncoder()
lengths=enc.fit_transform(lengths.reshape(-1,1)).toarray()
digits=enc.fit_transform(digits.reshape(-1,1)).toarray()

digit1=[]
digit2=[]
digit3=[]
digit4=[]
digit5=[]

for i in range(images_used):
	digit1.append(digits[i*5])
	digit2.append(digits[i*5+1])
	digit3.append(digits[i*5+2])
	digit4.append(digits[i*5+3])
	digit5.append(digits[i*5+4])

digit1=np.asarray(digit1)
digit2=np.asarray(digit2)
digit3=np.asarray(digit3)
digit4=np.asarray(digit4)
digit5=np.asarray(digit5)

net_output=[lengths,digit1,digit2,digit3,digit4,digit5]
print(net_output)

for i in range(images_used):
	image_name=str(i+1)+'.png'
	image=Image.open(image_path+image_name)
	train_dataset.append(np.array(image))
	
train_dataset=.5*(train_dataset-np.mean(train_dataset))/np.std(train_dataset)

print(train_dataset)
