print("Intializing.....")
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.utils import np_utils
print("Done!")
print("")

learning_rate = 0.001
epoch = 20
batch_size = 100

def split_data(X, y, test_data_size):
    return train_test_split(X, y, test_data_size, random_state=42, shuffle=False)

def reshape_data(arr, img_row, img_col, channels):
    return arr.reshape(arr.shape[0], img_row, img_col, channels)

ROOT_PATH = "E:\\FOR UNIVERSITY\\Special-Document\\Graduation-Project\\Images\\"
img_rows, img_cols = 128, 128
channels = 1

print("Loading dataset.....")
labels = pd.read_csv(ROOT_PATH + 'Data_samples2.csv')
y = labels['Finding Labels']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = y.reshape(-1, 1) 
X = np.load(ROOT_PATH + 'X_train2.npy')
print("Done!")
print("")

print("Splitting data into testing/training datasets.....")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
print("Done!")
print("")

print("Reshaping Data.....")
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)
input_shape = (img_rows, img_cols, channels)
print("Done!")
print("")

print("Normalizing Data.....")
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print("Done!")
print("")

X_train /= 255
X_test /= 255

training_iters=20
batch_size=100
learning_rate = 0.001
nb_input = 128
nb_classes = 15

print("Processing labels.....")
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)
print("Done!")
print("")


print("Initializing Placeholder.....")
x_place = tf.placeholder(tf.float32, [None, nb_input,nb_input,1])
y_place = tf.placeholder(tf.float32, [None, nb_classes])
print("x_place Shape: ", x_place.shape)
print("y_place Shape: ", y_place.shape)
print("Done!")
print("")

def conv2d(x_place, W, b, strides=1):
    x_place = tf.nn.conv2d(x_place, W, strides=[1, strides, strides, 1], padding='SAME')
    x_place = tf.nn.bias_add(x_place, b)
    return tf.nn.relu(x_place)
def maxpool2d(x_place, k=2):
    return tf.nn.max_pool(x_place, ksize = [1, k, k, 1], strides=[1,k,k,1], padding='SAME')

print("Building Model.....")
W = {
    'wc1': tf.get_variable('Wa0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('Wa1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('Wa2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('Wa3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('Wa6', shape=(128,nb_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
b = {
    'bc1': tf.get_variable('Ba0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('Ba1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('Ba2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('Ba3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('Ba4', shape=(nb_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x_place, W, b):
    #conv1 = conv2d(x_place, W['wc1'], b['bc1'])
    #conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.layers.conv2d(x_place, 32, 3, activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
    
    #conv2 = conv2d(conv1, W['wc2'], b['bc2'])
    #conv2 = maxpool2d(conv2, k = 2)
    #conv3 = conv2d(conv2, W['wc3'], b['bc3'])
    #conv3 = maxpool2d(conv3, k=2)
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
    conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
    conv4 = tf.layers.conv2d(conv3, 256, 3, activation=tf.nn.relu)
    conv4 = tf.layers.max_pooling2d(conv4, 2, 2)
    
    #fc1 = tf.reshape(conv3, [-1, W['wd1'].get_shape().as_list()[0]])
    #fc1 = tf.add(tf.matmul(fc1, W['wd1']), b['bd1'])
    #fc1 = tf.nn.softmax(fc1)
    fc1 = tf.contrib.layers.flatten(conv4)
    fc1 = tf.layers.dense(fc1, 4096)
    fc1 = tf.layers.dropout(fc1, rate=0.5)
    
    #out = tf.add(tf.matmul(fc1, W['out']),b['out'])
    out = tf.layers.dense(fc1, 15)
    return out

# Loss and Optimizer Node
pred = conv_net(x_place, W, b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y_place))
    # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)):
    # calculate the softmax cross entropy logits and labels, after that, calculate the mean of 1-D output array
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Optimize the minimum cost with learning rates
    
# Show the result
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_place, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess, tf.device('/device:GPU:0'):
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accu = []
    test_accu = []
    temp_accu = []
    temp_valid_loss = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph) # tf.summary.FileWriter: write event to a file
    for i in range(training_iters):
        for batch in range(len(X_train)//batch_size):
            batch_x = X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
            batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x_place: batch_x, y_place: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x_place: batch_x, y_place: batch_y})
        print("Iter " + str(i))
        print("Loss= " + "{:.2f}".format(loss) + ", Training Accuracy= " + "{:.2f}%".format(acc*100))
        print("Optimization Finished!")
        
        correct = 0
        for step in range(int(X_test.shape[0]/batch_size)):
            offset = step + batch_size
            batch_x_test = X_test[offset:(offset+batch_size), :, :, :]
            batch_y_test = y_test[offset:(offset+batch_size)]
            feed_dict = {x_place:batch_x_test, y_place:batch_y_test}
            test_acc,valid_loss = sess.run([accuracy,cost], feed_dict=feed_dict)
        temp_accu.append(test_acc)
        acc_result = sum(temp_accu)/len(temp_accu)
        temp_valid_loss.append(valid_loss)
        valid_loss_result = sum(temp_valid_loss)/len(temp_valid_loss)
        print("Validation loss:"+"{:.2f}%".format(valid_loss_result*100) + ", Testing Accuracy:"+"{:.2f}%".format(acc_result*100))
        #test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x_place: X_test,y_place : y_test})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accu.append(acc)
        test_accu.append(test_acc)
        print("Iter " + str(i) + " completed!")
        print("")
    summary_writer.close()
    print("Done!")
    
