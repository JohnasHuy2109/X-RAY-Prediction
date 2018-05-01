import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_data(X, y, test_data_size):
    return train_test_split(X, y, test_data_size, random_state=42, shuffle=False)

def reshape_data(arr, img_row, img_col, channels):
    return arr.reshape(arr.shape[0], img_row, img_col, channels)

def shuffle(matrix, target, test_proportion):
    ratio = matrix.shape[0]/test_proportion
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

def cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, img_row, img_col):
    
    model = Sequential()
    
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(img_row, img_col, channels)))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    nb_filters = 64
    kernel_size = (4, 4)

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    nb_filters = 128
    kernel_size = (8, 8)

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(12, 12)))

    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    stop = EarlyStopping(monitor='acc',
                         min_delta=0.001,
                         patience=2,
                         verbose=0,
                         mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1,
              validation_split=0.2,
              class_weight='auto',
              callbacks=[stop, tensor_board]
              )
    return model

if __name__ == '__main__':
    
    ROOT_PATH = "E:\\FOR UNIVERSITY\\Special-Document\\Graduation-Project\\Images\\"
    batch_size = 100
    nb_classes = 15
    nb_epoch = 20
    img_rows, img_cols = 256, 256
    channels = 1
    nb_filters = 32
    kernel_size = (2, 2)

    labels = pd.read_csv(ROOT_PATH + 'Data_samples.csv')
    y = labels['Finding Labels']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = y.reshape(-1, 1) 
    X = np.load(ROOT_PATH + 'X_train.npy')
   
    print("Splitting data into test/ train datasets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print("Done!")

    print("Reshaping Data")
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
    print("Done!")

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    input_shape = (img_rows, img_cols, channels)

    print("Normalizing Data")
    n = len(X_train)
    m = len(X_test)
    d1 = X_train[:,:int(0.25*n)].astype('float32')
    d2 = X_train[:,int(0.25*n):int(0.5*n)].astype('float32')
    d3 = X_train[:,int(0.5*n):int(0.75*n)].astype('float32')
    d4 = X_train[:,int(0.75*n):].astype('float32')
    m1 = X_test[:,:int(0.25*m)].astype('float32')
    m2 = X_test[:,int(0.25*m):int(0.5*m)].astype('float32')
    m3 = X_test[:,int(0.5*m):int(0.75*m)].astype('float32')
    m4 = X_test[:,int(0.75*m):].astype('float32')
    X_train = np.hstack(d1, d2, d3, d4)
    X_test = np.hstack(m1, m2, m3, m4)
    print("Done!")

    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    print("Building Model......")
    model = cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, img_rows, img_cols)
    print("Done!")

    print("Predicting")
    y_pred = model.predict(X_test)

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Done!")