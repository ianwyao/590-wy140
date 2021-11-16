'''
Weihan Yao
Oct 17th, 2021
''' 

from keras import layers 
from keras import models
import numpy as np
import warnings
from matplotlib import pyplot as plt
from random import randrange
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from skimage.transform import rescale, resize, downscale_local_mean
from keras import models
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import pickle
warnings.filterwarnings("ignore")

#HYPER PARAMETERS
flag = 'mnist'
network = 'cnn'
NKEEP=10000
batch_size=int(0.05*NKEEP)
epochs=20
rate=0.8
opt='rmsprop'
los='categorical_crossentropy'
met=['accuracy']
data_augment = False

#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------

#-------------------------------------
#GET 3 DATA AND REFORMAT
#-------------------------------------

#MNIST, FASHION-MNIST, CIFAR10
if flag == 'mnist':
    if network == 'ann':
        model = models.Sequential()
        model.add(layers.Dense(256, input_shape=(784,), activation="sigmoid"))
        model.add(layers.Dense(128, activation="sigmoid"))
        model.add(layers.Dense(10, activation="softmax"))
      
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape((60000, 784))
        test_images = test_images.reshape((10000, 784))
    
        #NORMALIZE
        train_images = train_images.astype('float32') / 255 
        test_images = test_images.astype('float32') / 255  
        
    if network == 'cnn':
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        
        model.summary()
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
    
        #NORMALIZE
        train_images = train_images.astype('float32') / 255 
        test_images = test_images.astype('float32') / 255  

if flag == 'fashion_mnist':
    if network == 'ann':
        model = models.Sequential()
        model.add(layers.Dense(256, input_shape=(784,), activation="sigmoid"))
        model.add(layers.Dense(128, activation="sigmoid"))
        model.add(layers.Dense(10, activation="softmax"))
      
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images.reshape((60000, 784))
        test_images = test_images.reshape((10000, 784))
    
        #NORMALIZE
        train_images = train_images.astype('float32') / 255 
        test_images = test_images.astype('float32') / 255  
        
    if network == 'cnn':
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(10, activation='softmax'))
        
        model.summary()
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        
        #NORMALIZE
        train_images = train_images.astype('float32') / 255 
        test_images = test_images.astype('float32') / 255  

if flag == 'cifar10':
    if network == 'ann':
        model = models.Sequential()
        model.add(layers.Dense(256, input_shape=(32*32*3,), activation="sigmoid"))
        model.add(layers.Dense(128, activation="sigmoid"))
        model.add(layers.Dense(10, activation="softmax"))
      
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.reshape((50000, 32*32*3))
        test_images = test_images.reshape((10000, 32*32*3))
    
        #NORMALIZE
        train_images = train_images.astype('float32') / 255 
        test_images = test_images.astype('float32') / 255  
        
    if network == 'cnn':
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(10, activation='softmax'))
        
        model.summary()
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
        train_images = train_images.reshape((50000, 32, 32, 3))
        test_images = test_images.reshape((10000, 32, 32, 3))
        
        #NORMALIZE
        train_images = train_images.astype('float32') / 255 
        test_images = test_images.astype('float32') / 255  


#DEBUGGING
print("batch_size",batch_size)
rand_indices = np.random.permutation(train_images.shape[0])
temp1 = train_images
temp2 = train_labels
if network == 'cnn':
    train_images=temp1[rand_indices[0:int(NKEEP*rate)],:,:]
    train_labels=temp2[rand_indices[0:int(NKEEP*rate)]]
    validation_images=temp1[rand_indices[int(NKEEP*rate):NKEEP],:,:]
    validation_labels=temp2[rand_indices[int(NKEEP*rate):NKEEP]]
if network == 'ann':
    train_images=temp1[rand_indices[0:int(NKEEP*rate)]]
    train_labels=temp2[rand_indices[0:int(NKEEP*rate)]]
    validation_images=temp1[rand_indices[int(NKEEP*rate):NKEEP]]
    validation_labels=temp2[rand_indices[int(NKEEP*rate):NKEEP]]

# exit()

#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
validation_labels = to_categorical(validation_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)

#Data Augmentation
if data_augment == True:
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range= 10,
        fill_mode='nearest',
        validation_split = 0.2
        )
    
    # generate new train/validation data
    model.compile(loss=los, optimizer=opt, metrics=met)
    datagen.fit(train_images)
    train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size, subset='training')
    validation_generator = datagen.flow(train_images, train_labels, batch_size=batch_size, subset='validation')
    
    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        steps_per_epoch = len(train_generator),
                        validation_steps = len(validation_generator),
                        epochs = epochs,
                        workers=-1)


#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
if data_augment == False:
    model.compile(optimizer=opt,
                  loss=los,
                  metrics=met)
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,validation_data = (validation_images,validation_labels))


#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)

#SAVE THE MODEL AND PARAMETERS
def save_model():
    # save model and architecture to single file
    model.save("model.h5")
    print("Saved model to disk")
    f = open('parameters.pckl', 'wb')
    pickle.dump([flag,network,NKEEP,batch_size,epochs,rate,opt,los,met,data_augment], f)
    f.close()

#LOAD THE MODEL AND PARAMETERS
def load_model():
    with open('parameters.pckl','rb') as f:
        flag,network,NKEEP,batch_size,epochs,rate,opt,los,met,data_augment = pickle.load(f)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
 
#SHOW A RANDOM IMAGE
ind = randrange(len(train_images))
image=train_images[ind]
#print((255*image).astype(int))
plt.imshow(image, cmap=plt.cm.gray); plt.show()

#Instantiating a model from an input tensor and a list of output tensors
layer_outputs = [layer.output for layer in model.layers[:5]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(train_images)

#First convolution layer
first_layer_activation = activations[0]
print(first_layer_activation.shape)

#Visualizing the fourth channel
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

#Visualizing every channel
layer_names = []
for layer in model.layers[:5]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
            :, :,
            col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
            row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
    scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
    
#Plot history
#Accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0,0.1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,0.1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()