
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization,SpatialDropout1D,Bidirectional, Embedding, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import re
import json

np.set_printoptions(threshold=sys.maxsize)

pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)


# # # Voor GPU support
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Inlezen van de dataset

dataset = pd.read_csv('./Dataset/mushrooms.csv')

# Print de eerste 5 lijnen van de dataset
dataset.head()

# Controleren op null values
dataset.isnull().sum()

# Momenteel bestaat de data uit verschillende letters per kolom wat moeilijk te begrijpen is
# Met de get_dummies function zorgen we ervoor dat we onze kolommen one-hot encoden
data_dum = pd.get_dummies(dataset)

# Print de shape van de one-hot encoded data
data_dum.shape

# Print de eerste 5 lijnen van de one-hot encoded data
# Hierin kunnen we zien dat we nu veel meer kolommen hebben en de letters weg zijn
data_dum.head()

# Controleren of de dataset gebalanceerd is
sns.countplot(x="class", data=dataset)

# Definiëer X en Y

# In onze x_data willen we alle data behalve of de paddenstoel eetbaar is of niet dus nemen we alle kolommen na deze 2
# We zetten dit ook om naar een numpy-array met type float
X_data = data_dum.loc[:, 'cap-shape_b':].to_numpy().astype(np.float32)


# In onze y_data mogen we enkel de klasses p en e (poisonous & edible) hebben
# We zetten dit ook om naar een numpy-array met type float
y_data = data_dum.loc[:, :'class_p'].to_numpy().astype(np.float32)

# We definiëren dat de trainingset 5000 samples moet zijn
n = 5000

# Neem de eerste 5000 samples van de x_data en steek deze in X_train
X_train = X_data[:n]

# Onze testset X_test krijgt de overige ~3000 samples
X_test = X_data[n:]

# Neem de eerste 5000 samples van de y_data en steek deze in y_train
y_train = y_data[:n]

# Onze testset y_test krijgt de overige ~3000 samples
y_test = y_data[n:]

# Print de shapes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Definiëer de INPUT_SIZE met het aantal kolommen dat X_train heeft
INPUT_SIZE = X_train.shape[1]

# Definiëer de OUTPUT_SIZE met het aantal kolommen dat y_train heeft
OUTPUT_SIZE = y_train.shape[1]

# Definiëer EPOCHS_NUM, het aantal epochs dat het model moet doorlopen
EPOCH_NUM = 50

# Definiëer de BATCH_SIZE dat het model moet gebruiken
BATCH_SIZE = 256

# Maak een placeholder aan met het datatype en de shape voor de inputs
inputs = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])

# Maak een placeholder aan met het datatype en de shape voor de outputs
outputs = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

# Maak een nieuwe variabele waarin we een truncated value maken om saturatie te voorkomen waardoor de neuroon niet meer zou bijleren
w = tf.Variable(tf.truncated_normal([INPUT_SIZE, OUTPUT_SIZE], stddev=0.1), dtype=tf.float32)

# Maak een nieuwe variabele waarin we een constante tensor aanmaken
b = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]), dtype=tf.float32)

# Bereken de y_pred
y_pred = tf.matmul(inputs, w) + b

# Maak de de loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs, logits=y_pred))

# Laat de Adam Optimizer de train step aanpassen
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# Definiëer hoe de juiste prediction gevonden wordt
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(outputs, 1))

# Definiëer hoe de accuracy berekend wordt
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Maak een nieuwe Session
sess = tf.Session()

# Maak een saver om het model op te slaan
saver = tf.train.Saver()

# Run de session met de globale variabelen
sess.run(tf.global_variables_initializer())

# Blijf de for-loop uitvoeren tot het meegegeven aantal epochs is uitgevoerd
for epoch in tqdm(range(EPOCH_NUM), file=sys.stdout):
    # Maak een willekeurige permutatie in onze range van 5000 (n)
    perm = np.random.permutation(n)

    prev_test_acc = 0

    for i in range(0, n, BATCH_SIZE):
        # Definiëer de X_batch via de eerder gekozen permutatie
        X_batch = X_train[perm[i:i+BATCH_SIZE]]

        # Definiëer de y_batch via de eerder gekozen permutatie
        y_batch = y_train[perm[i:i+BATCH_SIZE]]

        # Run de train (adam)
        train_step.run(session=sess, feed_dict={inputs: X_batch, outputs: y_batch})
    
    # Evalueer de accuracy op de training data
    acc = accuracy.eval(session=sess, feed_dict={inputs: X_train, outputs: y_train})

    # Evalueer de accuracy op de test data
    test_acc = accuracy.eval(session=sess, feed_dict={inputs: X_test, outputs: y_test})

    # Print de accuracy en validation accuracy elke epoch
    if (epoch+1) % 1 == 0:
        tqdm.write('epoch:\t%i\taccuracy:\t%f\tvalidation accuracy:\t%f' % (epoch+1, acc, test_acc))

        # Definiëer het path naar waar het model gesaved moet worden en save het
        if (test_acc > prev_test_acc):
            save_path = saver.save(sess, './Models/model.ckpt')
            print("Model saved in path: %s" % save_path)
            prev_test_acc = test_acc

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=5000, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(117,)),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
	tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# pas earlystopping toe
earlystopper = EarlyStopping(patience=20, verbose=1)

# sla het beste model op
checkpointer = ModelCheckpoint('./Models/model_keras.h5', verbose=1, save_best_only=True)

# train het model
history = model.fit(X_train, y_train, epochs=10, verbose=1, batch_size=256, validation_data=(X_test, y_test), callbacks=[earlystopper, checkpointer])

# Plot of the training history

# Accuracy
plt.plot(history.history['acc'],'r')
plt.plot(history.history['val_acc'],'b')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Loss 
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'b')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

rounded_labels=np.argmax(y_test, axis=1)

# Testen met de test set
# model = tf.keras.models.load_model('./Models/model_keras.h5')

y_pred = model.predict_classes(X_test)
print('\n')
print(classification_report(rounded_labels, y_pred))
print('\n')
cf = confusion_matrix(rounded_labels, y_pred)
print(accuracy_score(rounded_labels, y_pred) * 100)

y_pred = model.predict(X_test)
print('MAE: %f' % (mean_absolute_error(y_pred, y_test)))
print('R2: %f' % (r2_score(y_pred, y_test))) 

test_array_poison = np.array([[1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0]])

test_array_edible = np.array([[0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0]])

print(test_array_poison.shape)
print(test_array_edible.shape)


model.predict(test_array_poison)

model.predict(test_array_edible)

model.predict(test_array_edible)