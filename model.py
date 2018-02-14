import numpy as np
import tensorflow as tf

def test(testData, testLabels, classifier):
    batchsize=50
    correct=0.
    for data,label in DataBatch(testData,testLabels,batchsize):
        prediction = classifier(data)
        #print (prediction)
        correct += np.sum(prediction==label)
    return correct/testData.shape[0]*100

def DataBatch(data, label, batchsize, shuffle=True):
    n = data.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield data[inds], label[inds]

class CNNClassifier():
    def __init__(self, margin=32, classes=2, n=16, n_channel=4, model_saved='', k=12):
        
        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, shape=[None, 2*margin, 2*margin, n_channel]) # input batch of images
        self.y_ = tf.placeholder(tf.int64, shape=[None]) # input labels
        
        comp1 = self.composition_layer(self.x, k, 8)
        comp2 = self.composition_layer(tf.concat([self.x, comp1], axis=-1), k, 4)
        comp3 = self.composition_layer(tf.concat([self.x, comp1, comp2], axis=-1), k, 2)
        feats = tf.concat([self.x, comp1, comp2, comp3], axis=-1)
        trans = self.transition_layer(feats, k)

        comp4 = self.composition_layer(trans, k, 8)
        comp5 = self.composition_layer(tf.concat([trans, comp4], axis=-1), k, 4)
        comp6 = self.composition_layer(tf.concat([trans, comp4, comp5], axis=-1), k, 2)
        output = tf.concat([trans, comp4, comp5, comp6], axis=-1)
        fc1 = tf.contrib.layers.flatten(output)
        self.y = tf.layers.dense(fc1, classes)

        '''
        conv1 = tf.layers.conv2d(self.x, 16, 7, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        
        conv2 = tf.layers.conv2d(conv1, 32, 5, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        
        conv3 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
        
        fc1 = tf.contrib.layers.flatten(conv3)
        self.y = tf.layers.dense(fc1, classes)
        '''
        
        self.prediction = tf.argmax(self.y, axis=1)
        self.prob = tf.exp(self.y) / tf.reduce_sum(tf.exp(self.y), axis=1)
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(self.prediction, self.y_)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.saver = tf.train.Saver()
        
        if model_saved != '':
            self.load_model(model_saved)

    def composition_layer(self, x, n_kernel, kernel_size):
        BN1 = tf.layers.batch_normalization(x)
        ReLU1 = tf.nn.relu(BN1)
        conv1 = tf.layers.conv2d(ReLU1, n_kernel, kernel_size, padding='same', activation=None)
        return conv1
        #BN3 = tf.layers.batch_normalization(conv1)
        #ReLU3 = tf.nn.relu(BN3)
        #conv3 = tf.layers.conv2d(ReLU3, 3, n_kernel, activation=None)
    
    def transition_layer(self, x, n_kernel):
        conv = tf.layers.conv2d(x, n_kernel, 1, activation=None)
        pool = tf.layers.max_pooling2d(conv, pool_size=2, strides=2)
        return pool

    
    def load_model(self, model_saved):
        tf.reset_default_graph()
        self.saver.restore(self.sess, model_saved)
    
    def save_model(self):
        self.saver.save(self.sess, "phone_finder_CNN.ckpt")
            
    def train(self, trainData, trainLabels, devData, devLabels, epochs=1, batchsize=50):
        self.sess.run(tf.global_variables_initializer())
        
        max_acc = 97
        for epoch in range(epochs):
            total_cost = 0.0
            for i, (data,label) in enumerate(DataBatch(trainData, trainLabels, batchsize, shuffle=True)):
                _, hx = self.sess.run([self.train_step, self.cross_entropy], 
                                      feed_dict={self.x: data, self.y_: label})
                total_cost += hx
            dev_acc = test(devData, devLabels, self)
            
            print ('testing epoch:%d cost: %f accuracy: %f'%(epoch+1, total_cost, dev_acc))
            if dev_acc > max_acc:
                max_acc = dev_acc
                self.saver.save(self.sess, "./phone_finder_CNN.ckpt")
                
            #if dev_acc > 99.9:
            #    break
    
    def proba(self, x):
        return self.sess.run(self.prob, feed_dict={self.x: x})
        
    def __call__(self, x):
        return self.sess.run(self.prediction, feed_dict={self.x: x})

