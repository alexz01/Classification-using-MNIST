import MLP 
import CNN
import LoadUSPS
from logistic import LogisticRegression 
import numpy as np

from mnist import MNIST
mndata = MNIST('./MNIST')
trImg, trLab = mndata.load_training()
teImg, teLab = mndata.load_testing()

trImg = np.asanyarray(trImg)
trLab = np.asanyarray(trLab)
teImg = np.asanyarray(teImg)
teLab = np.asanyarray(teLab)

usps = LoadUSPS.LoadUSPS('proj3_images.zip')
uspsImg, uspsLab = usps.load()

#1> logistic Regression
logistic = LogisticRegression(28 * 28, 10)
logistic.train(trImg, trLab, lr = 0.3)
accuracy = logistic.test(teImg, teLab)
uspsacc = logistic.test(uspsImg, uspsLab)
print('logisticregression accuracy :', accuracy, uspsacc)

#grid search for best learning rate performance 
#for lr in [0.5, 0.3, 0.1, 0.05, 0.01]:
#    logistic.train(trImg, trLab, lr = 0.1)
#    accuracy = logistic.test(teImg, teLab)
#    print(lr, accuracy)



#2> Multilayer perceptron implementation using tensorflow
mlp = MLP.MLP()
mlp.train()

#grid search for best learning rate performance 
#for lr in [0.5, 0.3, 0.1, 0.05, 0.01]:
#   for node i [300, 450, 500, 800]:
#       mlp = MLP.MLP(nodes=node, lrate=lr)
#       mlp.train()
#Defaults set in Classes are best found during search

#3> Convolution Neural Network implemented taking help from the code provided in TA Slide
cnn = CNN.CNN()
cnn.train()

#grid search for best learning rate performance 
#for lr in [0.5, 0.3, 0.1, 0.05, 0.01]:
#   cnn = CNN.CNN(lrate=lr)
#   cnn.train()
#Defaults set in Classes are best found during search

