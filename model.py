'''Building training set'''
import os
import csv
from scipy import ndimage
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


'''Get data from csv file'''
lines=[]
#imgs=[]#images(X_train)#if not using generator
#steer=[]#steering values(y_train)#if not using generator
sideDelta=.2#how much to adjust steering for left and right images
with open('./data/driving_log.csv') as dataFile:
    reader=csv.reader(dataFile)
    next(reader,None)#skip headers
    for line in reader:
        lines.append(line)
        '''centSteer=float(line[3])#if not using generator
        for i in range(3):
            img=ndimage.imread('./data/IMG/'+line[i].split('/')[-1])
            imgs.append(img)
            imgs.append(np.fliplr(img))#flipped image
            if i==2:#right img
                steer.append(centSteer-sideDelta)
                steer.append(sideDelta-centSteer)#steering *-1 for flipped image
            else:#left and center imgs
                steer.append(centSteer+i*sideDelta)
                steer.append(-centSteer-i*sideDelta)#steering *-1 for flipped image
X_train=np.array(imgs)
y_train=np.array(steer)'''

'''Generator to use to get the batches of data'''
def gen(samples,batchSize=32):#really batch size =32*6 since each time it gets 6 images
    numSamps=len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for off in range(0,numSamps,batchSize):
            batchSamps=samples[off:off+batchSize]

            imgs=[]#images(X_train)
            steer=[]#steering values(y_train)
            for line in batchSamps:
                img=ndimage.imread('./data/IMG/'+line[0].split('/')[-1])
                imgs.append(img)
                steer.append(float(line[3]))
                centSteer=float(line[3])
                for i in range(3):
                    img=ndimage.imread('./data/IMG/'+line[i].split('/')[-1])
                    imgs.append(img)
                    imgs.append(np.fliplr(img))#flipped image
                    if i==2:#right img
                        steer.append(centSteer-sideDelta)
                        steer.append(sideDelta-centSteer)#steering *-1 for flipped image
                    else:#left and center imgs
                        steer.append(centSteer+i*sideDelta)
                        steer.append(-centSteer-i*sideDelta)#steering *-1 for flipped image'''
            X_train=np.array(imgs)
            y_train=np.array(steer)
            yield sklearn.utils.shuffle(X_train, y_train)

'''Split samples int training and validation sets and make generators for each'''
trainSamps,validSamps=train_test_split(lines,test_size=.2)
trainGen=gen(trainSamps)
validGen=gen(validSamps)

'''Model'''
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Conv2D,MaxPooling2D,Dropout

'''Model architecture using one found in NVIDIA paper(http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)'''
model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))#crop out unecessary parts of images(take 75 pixels from top, 25 from bottom, 0 from each side)
model.add(Lambda(lambda x:x/255.0-.5))#normalize and try to center mean around 0
model.add(Conv2D(24,5,strides=(2,2),activation='relu'))
model.add(Conv2D(36,5,strides=(2,2),activation='relu'))
model.add(Conv2D(48,5,strides=(2,2),activation='relu'))
model.add(Conv2D(64,3,activation='relu'))
model.add(Conv2D(64,3,activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))#,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50))#,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10))#,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

#TODO:Let me stop and save model at anytime(if i press a key then it will save model then exit)
'''Train model'''
model.compile(loss='mse',optimizer='adam')#loss function and optimizer to use
#model.fit(X_train,y_train,validation_split=.2,shuffle=True,epochs=5)#no generator
model.fit_generator(trainGen,steps_per_epoch=len(trainSamps)/32,validation_data=validGen,validation_steps=len(validSamps)/32,epochs=2)#train model with given settings

'''Save trained model and exit'''
model.save('model.h5')
exit()
