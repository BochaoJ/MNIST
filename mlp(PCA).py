import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
import time



def build_multilayer_perceptron(dim):
    model = Sequential()

    model.add(Dense(100, input_shape=(dim,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(90))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model






np.random.seed(222)

print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")


data = dataset.data     # dataset.data.shape=(70000,784)
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.target.astype("int"), test_size=10000)

print(trainData.shape)
print(testData.shape)


trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

#print trainLabels.shape
#print testLabels.shape


Data_train, Data_validation, Labels_train, Labels_validation = train_test_split(trainData,trainLabels,test_size=10000)

lab_train = Labels_train.argmax(axis=1)
lab_test = testLabels.argmax(axis=1)
lab_valid = Labels_validation.argmax(axis=1)
plt.subplot(131)
plt.hist(lab_train)
plt.title('Train Dataset')
plt.ylabel('Frequency')
plt.xlabel('Labels')
plt.subplot(132)
plt.hist(lab_valid)
plt.title('Validation Dataset')
plt.xlabel('Labels')
plt.subplot(133)
plt.hist(lab_test)
plt.title('Test Dataset')
plt.xlabel('Labels')
# show the plot
plt.show()

image_train = (Data_train * 255).astype("uint8")
image_train=image_train.reshape(50000,28,28)

# plot 8 images as gray scale
plt.subplot(241)
plt.imshow(image_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(image_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(image_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(image_train[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(image_train[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(image_train[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(image_train[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(image_train[7], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()




# PCA

mean=np.mean(Data_train, axis = 0)
Data_train -= mean # zero-center the data (important)
cov_mat = np.dot(Data_train.T, Data_train)/ Data_train.shape[0] # get the data covariance
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

## check PCA ###

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(784), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(784), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

U,S,V = np.linalg.svd(cov_mat) # dim(U)=784*784
n=100  # dimension reduce from 784 to n
Data_train = np.dot(Data_train, U[:,:n])
#print Data_train.shape

mean=np.mean(Data_validation, axis = 0)
Data_validation -= mean
Data_validation = np.dot(Data_validation, U[:,:n])
#print Data_validation.shape

mean=np.mean(testData, axis = 0)
testData-= mean
testData = np.dot(testData, U[:,:n])

#print testData.shape



# plot the image after PCA
image=np.dot(Data_train,(np.transpose(U)[:n,:]))
image = (image * 255).astype("uint8")
image=image.reshape(50000,28,28)


# plot 8 images as gray scale
plt.subplot(241)
plt.imshow(image[0], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(image[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(image[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(image[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(image[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(image[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(image[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(image[7], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()



# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)    # 2: SGD(lr=0.01, momentum=0.9); 3(if RMS): opt=RMSprop()

model_SGD = build_multilayer_perceptron(n)

model_SGD.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

t1=time.time()
print("[INFO] training...")
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
mlp_SGD = model_SGD.fit(Data_train, Labels_train, validation_data=(Data_validation,Labels_validation), callbacks=[early_stopping],
			  batch_size=128, nb_epoch=20, verbose=1,shuffle=True)  ########## epoch
t2=time.time()
print(t2-t1)
# initialize the optimizer and model

print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9)    # 2: SGD(lr=0.01, momentum=0.9); 3(if RMS): opt=RMSprop()

model_Mom = build_multilayer_perceptron(n)

model_Mom.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

t1=time.time()
print("[INFO] training...")

mlp_mom=model_Mom.fit(Data_train, Labels_train, validation_data=(Data_validation,Labels_validation), callbacks=[early_stopping],
			  batch_size=128, nb_epoch=20, verbose=1,shuffle=True)  ########## epoch
t2=time.time()
print(t2-t1)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = RMSprop()   # 2: SGD(lr=0.01, momentum=0.9); 3(if RMS): opt=RMSprop()

model_RMS = build_multilayer_perceptron(n)

model_RMS.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


print("[INFO] training...")

mlp_RMS=model_RMS.fit(Data_train, Labels_train, validation_data=(Data_validation,Labels_validation), callbacks=[early_stopping],
			  batch_size=128, nb_epoch=20, verbose=1,shuffle=True)  ########## epoch

# list all data in history
# print(mlp_SGD.history.keys())
# summarize mlp_SGD for accuracy
plt.plot(mlp_SGD.history['acc'],'r--')
plt.plot(mlp_SGD.history['val_acc'],'b--')
plt.plot(mlp_mom.history['acc'],'r-.')
plt.plot(mlp_mom.history['val_acc'],'b-.')
plt.plot(mlp_RMS.history['acc'],'r')
plt.plot(mlp_RMS.history['val_acc'],'b')
plt.title('Model Accuracy for MLP Method')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['SGD-train', 'SGD-valid','Mom-train', 'Mom-valid','RMS-train', 'RMS-valid'], loc='lower right')
plt.show()
# summarize history for loss
plt.plot(mlp_SGD.history['loss'],'r--')
plt.plot(mlp_SGD.history['val_loss'],'b--')
plt.plot(mlp_mom.history['loss'],'r-.')
plt.plot(mlp_mom.history['val_loss'],'b-.')
plt.plot(mlp_RMS.history['loss'],'r')
plt.plot(mlp_RMS.history['val_loss'],'b')
plt.title('Learning Curves for MLP Method')
plt.ylabel('cross-entropy loss')
plt.xlabel('epoch')
plt.legend(['SGD-train', 'SGD-valid','Mom-train', 'Mom-valid','RMS-train', 'RMS-valid'], loc='upper right')
plt.show()


# show the accuracy on the testing set

probs = model_SGD.predict(testData)  # testData[np.newaxis, i].shape=(1, 10, 1, 28, 28)
prediction = probs.argmax(axis=1)
cm = confusion_matrix(testLabels.argmax(axis=1), prediction)
plt.matshow(cm)
plt.title('Confusion matrix for MLP-SGD')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

probs = model_Mom.predict(testData)  # testData[np.newaxis, i].shape=(1, 10, 1, 28, 28)
prediction = probs.argmax(axis=1)
cm = confusion_matrix(testLabels.argmax(axis=1), prediction)
plt.matshow(cm)
plt.title('Confusion matrix for MLP-Mom')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


probs = model_RMS.predict(testData)  # testData[np.newaxis, i].shape=(1, 10, 1, 28, 28)
prediction = probs.argmax(axis=1)
cm = confusion_matrix(testLabels.argmax(axis=1), prediction)
plt.matshow(cm)
plt.title('Confusion matrix for MLP-RMS')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


print("[INFO] evaluating...")
(loss, accuracy) = model_SGD.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


(loss, accuracy) = model_Mom.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

(loss, accuracy) = model_RMS.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))