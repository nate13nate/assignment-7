from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

improved = load_model('./models/improved.keras')
baseline = load_model('./models/keras.keras')
alexnet = load_model('./models/alexnet.keras')
lenet = load_model('./models/lenet.keras')
vgg16 = load_model('./models/keras-VGG16-cifar10.keras')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

sparse_y_train = y_train.reshape(-1,)

# Reshape converting 2D to 1D
sparse_y_test = y_test.reshape(-1,)
sparse_y_train = y_train.reshape(-1,)

# This code normalazation
sparse_x_train = X_train
sparse_x_test = X_test

improvedScores = improved.evaluate(X_test, y_test, verbose=0)
print("Improved Accuracy: %.2f%%" % (improvedScores[1]*100))

baselineScores = baseline.evaluate(X_test, y_test, verbose=0)
print("Baseline Accuracy: %.2f%%" % (baselineScores[1]*100))

# alexnetScores = alexnet.evaluate(sparse_x_test, sparse_y_test, verbose=1)
# print("Alexnet Accuracy: %.2f%%" % (alexnetScores[1]*100))

# lenetScores = lenet.evaluate(sparse_x_test, sparse_y_test, verbose=1)
# print("Lenet Accuracy: %.2f%%" % (lenetScores[1]*100))

vgg16Scores = vgg16.evaluate(X_test, y_test, verbose=0)
print("VGG 16 Accuracy: %.2f%%" % (vgg16Scores[1]*100))

y_predictions= improved.predict(X_test)
y_test.reshape(-1,)
y_predictions.reshape(-1,)
y_predictions= np.argmax(y_predictions, axis=1)
y_test= np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix, accuracy_score
plt.figure(figsize=(7, 6))
plt.title('Confusion matrix', fontsize=16)
plt.imshow(confusion_matrix(y_test, y_predictions))
plt.xticks(np.arange(10), classes, rotation=45, fontsize=12)
plt.yticks(np.arange(10), classes, fontsize=12)
plt.colorbar()
plt.show()

L = 8
W = 8
fig, axes = plt.subplots(L, W, figsize = (20,20))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i])
    axes[i].set_title("Predicted = {}\n Actual  = {}".format(classes[y_predictions[i]], classes[y_test[i]]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=1)
plt.show()
