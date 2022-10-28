import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
import matplotlib.image as mpimg
from keras.models import model_from_json

import seaborn as sns
from sklearn.metrics import confusion_matrix

file_number = 700
train_sum = int(file_number*4 / 5)
test_sum = int(file_number / 5)

def min_max(list, new_min=0, new_max=1):
    list_new = []
    for i in list:
        ii = ((i - np.min(list)) / (np.max(list) - np.min(list))) * (new_max - new_min) + new_min
        list_new.append(ii)
    return list_new

def insert(data,sum):
    newdata = []
    for i in range(658):
        step = (data[i+1]-data[i])/sum
        for j in range(sum):
            newdata = np.append(newdata,data[i]+j*step)
    return newdata

x_train = np.load('data/x_train.npy')

x_test = np.load('data/x_test.npy')

y_train = np.load('data/y_train.npy')

y_test = np.load('data/y_test.npy')

x_train, x_test = x_train / 2000.0, x_test / 2000.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=1, kernel_size=5, strides=1, padding='same'),
    tf.keras.layers.Conv1D(filters=1, kernel_size=5, strides=1, padding='same'),
    tf.keras.layers.Flatten(input_shape=(659, 1)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights('weights_94.93/')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='weights/',
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train,
                    epochs=0,
                    validation_data=(x_test, y_test),
                    verbose=2,
                    callbacks=[cp_callback])

x_100test = np.load('data/x_100test.npy')

x_100test = x_100test / 2000.0
pre = model.predict_classes(x_100test)
print(pre)

y_100test = np.load('data/y_100test.npy')

sns.set()
f,ax=plt.subplots(figsize=(8, 6.5))
C2= confusion_matrix(y_100test, pre, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])

print(C2)
sns.heatmap(C2,annot=True,ax=ax,fmt='.20g',cmap="YlGnBu")

ax.set_title('confusion matrix')
ax.set_xlabel('predict')
ax.set_ylabel('true')

i = 0
sum = 0
while i<15:
  sum = sum+C2[i,i]
  i = i+1

print(sum/15/100)

imp = np.zeros([15,659])
pre = model.predict(x_100test)

for k in range(15):
    k100 = 100*k
    for i in range(659):
        a = np.zeros([1500,659,1])
        a[:,:,:] = x_100test[:,:,:]
        if i<5:
            a[k100:k100+100, 0:i+5, :] = 0
            pre_1 = model.predict(a)
            imp[k,i] = np.average(pre[k100:k100+100,k]) - np.average(pre_1[k100:k100+100,k])
            print(i)
        else:
            a[k100:k100 + 100, i - 5:i + 5, :] = 0
            pre_1 = model.predict(a)
            imp[k, i] = np.average(pre[k100:k100 + 100, k]) - np.average(pre_1[k100:k100 + 100, k])
            print(i)

x = np.loadtxt('x.txt')

bac_name = ["MRSA","S. aureus","S. epidermidis","E. coli","B. subtilis","B. subtilis Var. Niger","E. faecalis","S. agalactiae","P. aeruginosa","S. paratgphi A","S. paratgphi B","C. albicans","C. glabrata","S. cerevisiae","R. rubra"]

x = insert(x,20)
for i in range(15):
    y = np.loadtxt('average_spectrum/{}.txt'.format(bac_name[i]))
    y = insert(y, 20)
    plt.clf()
    plt.figure(figsize=(15, 5))
    imp_ = min_max(imp[i,:])
    imp_ = insert(imp_, 20)
    imp_ = imp_*2/3
    imp_[1] = 1
    sc = plt.scatter(x, y, s=25, c=imp_, marker='o',cmap='cool')
    plt.colorbar(sc)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('fig/{}.png'.format(bac_name[i]))


