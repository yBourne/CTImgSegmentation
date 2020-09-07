#!/usr/bin/env python
# coding: utf-8



import cv2
import numpy as np

img = cv2.imread('./dataset/2_0051img.png') # 2_0051img.png
label = cv2.imread('./dataset/2_0051label.png') # 2_0051label.png
w_ind = 1
h_ind = 1
croped_img = np.zeros([128,128])
croped_label = np.zeros([128,128])
for i in range(8):
    h_ind += 128
    if  h_ind>img.shape[0]:
        h_ind = img.shape[0]
    for j in range(1,9):
        w_ind += 128
        if w_ind>img.shape[1]:
            w_ind = img.shape[1]
        croped_img = img[(h_ind-129):(h_ind-1), (w_ind-129):(w_ind-1)]
        croped_label = label[(h_ind-129):(h_ind-1), (w_ind-129):(w_ind-1)]
        cv2.imwrite('./dataset/croped-2/0051_{}img.png'.format(str(i*8+j)), croped_img)
        cv2.imwrite('./dataset/croped-2/0051_{}label.png'.format(str(i*8+j)), croped_label)
    w_ind = 1
            
print('dataset acquisition complete')


# keras中文文档- 图像预处理

# In[9]:



# 创建两个相同参数的实例
data_gen_args = dict(rotation_range=90,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip = True,)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# 为 fit 和 flow 函数提供相同的种子和关键字参数
seed = 1
# image_datagen.fit(images, augment=True, seed=seed)
# mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    './dataset/train_data_augment/imgs/',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    './dataset/train_data_augment/labels/',
    class_mode=None,
    seed=seed)

# 将生成器组合成一个产生图像和蒙版（mask）的生成器
train_generator = zip(image_generator, mask_generator)

for i in range(1):
    for img, label in train_generator:
        print(img, label)
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50)


# In[2]:


# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
img_t = load_img('./dataset/train_data_augment/0051_2label.png')
img_t = img_to_array(img_t)
np.unique(img_t[:,:,2]==img_t[:,:,1])


# In[6]:


np.unique(img_t[:,:,2])


# 图像融合和增强
# - 标号有问题

# In[6]:


# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
# import os, re
def data_gen(dire, to_dire):
    Datagen = ImageDataGenerator(rotation_range=40,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='constant')
    seed = 1
    for ind, file_name in enumerate(os.listdir(dire)):
        tp = os.path.splitext(file_name)[-1]
        if tp!='.png':
            break
        num = os.path.splitext(file_name)[0]
        num = re.search('[0-9]+_[0-9]+', num).group()
        if ind%2 == 0:
            img = load_img(dire+file_name)
            x_img = img_to_array(img)
        else:
            img = load_img(dire+file_name)
            y_img = img_to_array(img)
            x_img[:,:,1] = y_img[:,:,0]
            x_img[:,:,2] = y_img[:,:,1]
            x_img = x_img.reshape((1,)+ x_img.shape)
    #         x_img = x_img.reshape((1,)+ x_img.shape)
            i = 1
            for img_batch in Datagen.flow(x_img,
                          batch_size=32,
                          save_to_dir=to_dire,
    #                       seed = seed,
                          save_prefix = num+'_'+'img',
                          save_format='png'):
    #             print(img_batch.shape, label_batch)
                i +=1
                if i > 5:
                    break
    print('Augmentation complete!')

data_gen('./dataset/train_data_augment/orig-88/', './dataset/train_data_augment/augment-88/')


# 分离增强图像

# In[7]:


import cv2, re
def data_split(dire, to_dire):
    for ind, file_name in enumerate(os.listdir(dire)):
        tp = os.path.splitext(file_name)[-1]
        if tp!='.png':
            break
        num = os.path.splitext(file_name)[0]
        num = re.search('[0-9]+_[0-9]+_', num).group()
        img = load_img(dire + file_name)
        img = img_to_array(img)
        img_ = img.copy()
        img_[:,:,1] = img_[:,:,0]
        img_[:,:,2] = img_[:,:,0]
        img_label = img.copy()
        img_label[:,:,0] = img[:,:,1]
        img_label[:,:,1] = img[:,:,2]
        img_label[:,:,2] = 0
        img_.reshape((1,)+ img_.shape)
        img_label.reshape((1,)+ img_label.shape)
        save_img(to_dire+num+str(ind+1)+'img'+'.png', img_)
        save_img(to_dire+num+str(ind+1)+'label'+'.png', img_label)    
    print('Split complete!')
data_split('./dataset/train_data_augment/augment-88/', './dataset/train_data_augment/augment-88/split/')


# In[30]:


import keras
help(keras.preprocessing.image.save_img)


# In[17]:


import numpy as np

a = np.array([0,1,2,3,4,5])
print(label.shape)
cv2.imshow('check', label)
k = cv2.waitKey(0)
if k ==27:     # 键盘上Esc键的键值
    cv2.destroyAllWindows() 


# In[4]:


import time
import sys

for i in range(5):
    print(i, end='')
#     sys.stdout.flush()
    time.sleep(1)


# In[ ]:


from tensorflow.python.client import device_lib
import tensorflow as tf

print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())

