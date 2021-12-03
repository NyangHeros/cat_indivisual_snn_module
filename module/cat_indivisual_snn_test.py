# 2021-12-03 
# Modularization of predict part only
# referenced code : https://www.kaggle.com/jovi1018/cat-individual-snn

# development environment
# python version : 3.6.8
# needed library : tensorflow(version=2.6.2), numpy, matplotlib, sklearn, cv2, tqdm, os

# load model
import tensorflow as tf
from tensorflow.keras.models import load_model

# data visualisation and manipulation
import numpy as np
import matplotlib.pyplot as plt

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
from tqdm import tqdm
import os 

#dl libraraies
import tensorflow.keras.backend as K



class CatSnn:
 #---------------------------------initialation functions--------------------------------------
    def __init__(self,num_classes = 10,  img_size = 150,  weighs_path = "../weights/cat_indivisual_model_cl_weights.h5"):
        # num_classes: the number of cat individuals, selection: indivisuals chosen when learned, dataset_dir: dataset dir, weights_path: snn weights path

        # 추후 개선 사항 : dir, model_path 기본값 바꾸기.

        # self declaration
        self.num_classes = num_classes

        # standard image size, which will be sent to Neural Network
        self.IMG_SIZE = img_size

        loaded_base_model = self.embeddingModelCl((self.IMG_SIZE, self.IMG_SIZE, 3))
        self.model = self.completeModelCl(loaded_base_model)
        self.model.load_weights(weighs_path)
 #---------------------------------------------------------------------------------------



 #---------------------------------make functions--------------------------------------
    # function to automate store images of np.array() to X, and labels to Z 
    def makeTrainData(self, selection, dataset_dir, random_state):
        self.makeLabels(selection)

        # initial an empty list X to store image of np.array()
        self.X = []
        # initial an empty list Z to store labels/names of cat individauls
        self.Z = []

        for label in self.labels:
            DIR = dataset_dir + label
            for img in tqdm(os.listdir(DIR)):
                path = os.path.join(DIR,img)
                # reading images
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                # resizing images to (150, 150, 3), 3 is the number of channels - RGB
                img = cv2.resize(img, (self.IMG_SIZE,self.IMG_SIZE))
            
                self.X.append(np.array(img))
                self.Z.append(str(label))
        
     
        # # make train data
        # self.makeTrainData()

        ## Transform labels in Z to Y from class 0 to class 9, as 10 different cat individuals
        le=LabelEncoder()
        self.Y = None
        self.Y = le.fit_transform(self.Z)

        ## Transform and normalize X in the range of [0, 1]
        self.X=np.array(self.X)
        self.X=self.X/255.

        self.x_train,x_test,self.y_train,y_test=train_test_split(self.X, self.Y,test_size=0.2, stratify = self.Y, random_state=random_state)
        # x_val,x_test,y_val,y_test=train_test_split(x_test,y_test, stratify = y_test, test_size=0.5,random_state=random_state)

    def makeAnchorByTrainData(self):
        self.anchor_images = [self.x_train[self.y_train==i][0] for i in range(10)]
        self.anchor_images = np.array(self.anchor_images)

    def makeLabels(self, selection):
        self.labels=[]
        for i in selection:
            label = '000' + str(i)
            self.labels.append(label[-4:])

    def makeAnchor(self, selection, dataset_dir):
        # self.selection=selection
        # # directioary of storage of cat images
        # self.dir = dataset_dir
        ## initial an empty list 'labels' to store name of cat individual for each image.
        self.makeLabels(selection)

        imgs=[]

        for label in self.labels:
            DIR = dataset_dir + label
            img = os.listdir(DIR)[0]
            img_path = os.path.join(DIR,img)
            imgs.append(self.preprocessImage(img_path))

        self.anchor_images = np.array(imgs)
 #---------------------------------------------------------------------------------------



 #---------------------------------plot functions--------------------------------------
    def plotImage(this, img_path):
        img = this.preprocessImage(img_path)
        plt.imshow(img[0])

    def plotAnchor(self, names=[]):
        fig = plt.figure()
        rows = 4
        cols = 4

        if len(names)!=len(self.labels):
            names = self.labels

        for i, anchor in enumerate(self.anchor_images):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.imshow(anchor.reshape(1, 150, 150, 3)[0])
            ax.set_xlabel(names[i]+": %s" %self.dists[i])
            ax.set_xticks([]), ax.set_yticks([])

    def plotShow(this):
        plt.show()
 #---------------------------------------------------------------------------------------



 #---------------------------------predict functions--------------------------------------
    def preprocessImage(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.IMG_SIZE,self.IMG_SIZE))
        img = img/255.
        img = img.reshape(1, 150, 150, 3)
        return img

    def predict(self, image_path="../test_img/0006_022.JPG", selection = [6, 7, 15, 18, 19, 29, 55, 57, 82, 152], archer_dir="../archer_img/", dist_threshold=0.4):
        self.makeAnchor(selection, archer_dir)

        self.dists = []
        test_img = self.preprocessImage(image_path)
        for i in range(len(self.anchor_images)):
            anchor = self.anchor_images[i].reshape(1, 150, 150, 3)
            test_archor = [test_img, anchor]
            dist = self.model.predict(test_archor)
            self.dists.append(dist[0][0])
        self.dists = np.array(self.dists)
        return self.dists
 
    def predictId(self, image_path="../test_img/0006_022.JPG", selection = [6, 7, 15, 18, 19, 29, 55, 57, 82, 152], archer_dir="../archer_img/", dist_threshold=0.4):
        self.predict(image_path, selection, archer_dir, dist_threshold)
        if np.sum(self.dists <= dist_threshold) >= 1:
            idx = np.argmin(self.dists)
            return idx
        else:
            return -1
 #---------------------------------------------------------------------------------------



 #---------------------------------loss functions--------------------------------------
    # contrastive loss function
    def contrastiveLoss(self, y, preds, margin=1):
        # explicitly cast the true class label data type to the predicted
        # class label data type (otherwise we run the risk of having two
        # separate data types, causing TensorFlow to error out)
        y = tf.cast(y, preds.dtype)
        # calculate the contrastive loss between the true labels and
        # the predicted labels
        squaredPreds = K.square(preds)
        squaredMargin = K.square(K.maximum(margin - preds, 0))
        loss = K.mean((1 - y) * squaredPreds + y * squaredMargin)
        # return the computed contrastive loss to the calling function
        return loss

    # Function to calculate the distance between two images (Euclidean Distance used here)
    def euclideanDistance(self, vectors):
        # unpack the vectors into separate lists
        (featsA, featsB) = vectors
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                        keepdims=True)
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))
 #---------------------------------------------------------------------------------------



 #---------------------------------model functions--------------------------------------
    # Base model with pre-training VGG16
    def embeddingModelCl(self, inputShape, embeddingDim=128):
        # VGG16 as base_model
        base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                    input_shape = inputShape,
                                                    weights = 'imagenet')

        # freeze all the layers of VGG16, so they won't be trained.
        for layer in base_model.layers:
            layer.trainable = False
        
        inputs = tf.keras.layers.Input(shape=inputShape)
        #x = data_augmentation(inputs)
        x = base_model(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(units=embeddingDim)(x)
        
        model = tf.keras.Model(inputs, outputs)
        return model

    # Complete model
    def completeModelCl(self, base_model):
        # Create the complete model with pair
        # embedding models and minimize the distance for positive pair
        # and maximum the distance for negative pair
        # between their output embeddings
        imgA = tf.keras.layers.Input(shape=((self.IMG_SIZE, self.IMG_SIZE, 3)))
        imgB = tf.keras.layers.Input(shape=((self.IMG_SIZE, self.IMG_SIZE, 3)))

        base_model = self.embeddingModelCl((self.IMG_SIZE, self.IMG_SIZE, 3))
        
        featsA = base_model(imgA)
        featsB = base_model(imgB)
    
        distance = tf.keras.layers.Lambda(self.euclideanDistance)([featsA, featsB])
        model = tf.keras.Model(inputs=[imgA, imgB], outputs=distance)
        model.compile(loss=self.contrastiveLoss, optimizer="adam")
        return model
 #---------------------------------------------------------------------------------------