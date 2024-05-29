#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras


# In[3]:


IMG_HEIGHT=48 
IMG_WIDTH = 48
batch_size=32 #augments 32 images at a time 

train_data_dir=r"C:\Users\Karima moussa\OneDrive\Desktop\advancedproj\data\train"
validation_data_dir=r"C:\Users\Karima moussa\OneDrive\Desktop\advancedproj\data\val"

train_datagen = ImageDataGenerator( #this function tells me how to augment the data 
    rescale=1/255, #divide all pixels by 255
    rotation_range=40, #0-40% rotation
    width_shift_range=0.2, #shift image horizontally by 0-20%
    height_shift_range=0.2,
    shear_range=0.2, #shear image by 20%
    zoom_range=0.2, #zoom in and out by 20%
    horizontal_flip=True,) #th mirror of the image

validation_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40, #0-40% rotation
    width_shift_range=0.2, #shift image horizontally by 0-20%
    height_shift_range=0.2,
    shear_range=0.2, #shear image by 20%
    zoom_range=0.2, #zoom in and out by 20%
    horizontal_flip=True,) #th mirror of the image
 #we only neede to rescale because we need data in its original form 
#to make sure that the model is working when we validate using it 


# In[4]:


train_generator = train_datagen.flow_from_directory( # .we use flow from directory 
					train_data_dir,                               #because we have alot of images not only 1,this is the
					color_mode='grayscale',                            #function that applies the priv fun rules to the dataset
					target_size=(IMG_HEIGHT, IMG_WIDTH), #we want all images to be of the same size 
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True) ##ask tasneem 

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(IMG_HEIGHT, IMG_WIDTH),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


# In[ ]:





# In[5]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[ ]:


train_path=r"C:\Users\Karima moussa\OneDrive\Desktop\advancedproj\data\train"
test_path=r"C:\Users\Karima moussa\OneDrive\Desktop\advancedproj\data\test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files) #this function counts the number of files in my training data
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path): #this function counts the number of files in my test data
    num_test_imgs += len(files)

epochs=40
history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs,
                    batch_size=32,callbacks=[EarlyStopping(monitor='val_loss', patience=7, verbose=0)])


# In[ ]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']


plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[1]:


from keras.models import load_model

model.save('emotion_detection_model_100epochs.h5')
my_model = load_model('emotion_detection_model_100epochs.h5', compile=False)

test_img, test_lbl = validation_generator.__next__()
predictions=my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, predictions))


# In[53]:


class_labels=['Angry','Disgust', 'Fear', 'Happy','sad','surprised','neutral']
import random
n=random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.imshow(image[:,:,0], cmap='gray')
plt.title("Original label is:"+orig_labl+" Predicted is: "+ pred_labl)
plt.show()


# In[ ]:





# In[ ]:




