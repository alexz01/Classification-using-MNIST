import numpy as np

#import zipfile as zipf
#import scipy



class LogisticRegression:
    def __init__(self, dimension, classes):
        self.w = np.zeros((dimension, classes))
        self.b = np.zeros((1, classes))
        self.error = []

    def train(self, images, labels, lr):
        print('training')
        epoches=1000
        batch=500
        num_batches = int(images.shape[0] / batch)
        for epoch in range(epoches):
            indexs = np.random.permutation(images.shape[0])
            for batch in range(num_batches):
                batch_index = indexs[batch * batch:(batch + 1) * batch]
                x = images[batch_index, :]
                y = labels[batch_index]
                #softmax function
                e = np.exp(np.dot(x, self.w) + self.b - np.max(np.dot(x, self.w) + self.b, axis=1).reshape((-1, 1)))
                probability = e / np.sum(e, axis=1, keepdims=True)
                probability[range(batch), y] -= 1.0
                probability /= batch
#                err = np.sum(-np.log(probability[range(batch), y])) / batch
#                self.error.append(err)
                self.w += -lr * np.dot(x.T, probability)
                self.b += -lr * np.sum(probability, axis=0, keepdims=True)
            lr *= 0.9
        
    def test(self, images, labels):
        print('testing')
        probability = np.dot(images, self.w) + self.b
        prediction = np.argmax(probability, axis=1)
        return np.mean(prediction == labels)

#print('LogisticRegression Session')
#logistic = LogisticRegression(28 * 28, 10)
#logistic.train(trImg, trLab, lr = 0.3)
#accuracy = logistic.test(teImg, teLab)
#uspsacc = logistic.test(uspsImg, uspsLab)
#print('logisticregression accuracy :', accuracy, uspsacc)

#grid search for best learning rate performance 
#for lr in [0.5, 0.3, 0.1, 0.05, 0.01]:
#    logistic.train(trImg, trLab, lr = 0.1)
#    accuracy = logistic.test(teImg, teLab)
#    print(lr, accuracy)