#EXTRACTING DATA FROM PICKLED FILE
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

print("Reading the pickled file...")
datasets=[]
with (open("notMNIST.pickle","rb")) as openfile:
	while True:
		try:
			datasets.append(pickle.load(openfile))
		except EOFError:
			break


#print(datasets[0]['valid_dataset'].shape)
#print(datasets[0]['valid_labels'].shape)
#print(datasets[0]['valid_labels'][0:200])
'''
duplicate=0
test_image_num=0
n=10
#CHECK DUPLICATES FOR N NUMBER OF TRAIN IMAGES
print("Checking the duplicate data between training and test sets...")
for train_image in datasets[0]['train_dataset'][0:n]:
	print("At train image %s"%test_image_num)
	for test_image in datasets[0]['test_dataset']:
		if (train_image!=test_image).any()==False:
			duplicate=duplicate+1
	test_image_num=test_image_num+1	
print("Total duplicates %s"%duplicate)
'''
datasets[0]['train_dataset']=datasets[0]['train_dataset'][0:10000,:,:]
datasets[0]['test_dataset']=datasets[0]['test_dataset'][0:1000,:,:]
datasets[0]['train_labels']=datasets[0]['train_labels'][0:10000]
datasets[0]['test_labels']=datasets[0]['test_labels'][0:1000]
X_train=np.zeros((datasets[0]['train_dataset'].shape[0],28*28))
X_test=np.zeros((datasets[0]['test_dataset'].shape[0],28*28))
y_train=datasets[0]['train_labels']
y_test=datasets[0]['test_labels']
print("Flattening/Ravelling 2d pixel data into 1D...")
i=0
for train_image in datasets[0]['train_dataset']:
	X_train[i]=train_image.ravel()
	i=i+1
j=0
for test_image in datasets[0]['test_dataset']:
	X_test[j]=test_image.ravel()
	j=j+1


print("Performing simple logistic regression...")
lr=LogisticRegression(penalty="l1",C=.1,random_state=0)

lr.fit(X_train,y_train)

predict_alphabet=lr.predict(X_test)
#print(predict_alphabet)
#print(y_test)

print("Accuracy of alphebet prediction is %0.2f"%lr.score(X_test,y_test))



