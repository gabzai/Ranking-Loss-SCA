import os
import sys
import time
import pickle
import h5py
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn import preprocessing
from exploit_pred import *
from clr import OneCycleLR


### Scripts based on ASCAD-v1 github : https://github.com/ANSSI-FR/ASCAD

class Logger(object):
    def __init__(self, name_file):
        self.terminal = sys.stdout
        self.log = open(str(name_file)+".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

# Optimized implementation
# Credit: https://github.com/gabzai/Ranking-Loss-SCA/issues/1 
def ranking_loss_optimized(score_layer, alpha_value=10):
    def loss(y_true, y_pred):
        alpha = K.constant(alpha_value, dtype='float32')
        
        y_true = tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.int32)
        y_true_int = K.cast(y_true, dtype='int32')
        batch_s = K.cast(K.shape(y_true_int)[0], dtype='int32')

        # Indexing the training set (range_value = (?,))        
        range_value = K.arange(0, batch_s, dtype='int32')

        # Get rank and scores associated with the secret key (rank_sk = (?,))
        # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
        indices = tf.concat([range_value[:, None], y_true_int[:, None]], 1)
        score_sk = tf.gather_nd(score_layer, indices)

        # Ranking Loss sk*-sk
        loss_rank_total = score_sk[:, None]- score_layer
        
        # Ranking Logistic Loss sum_{k \in K/k*} log2(1 + exp(sk*-sk))
        logistic_loss = K.log(1 + K.exp(-1 * alpha * (loss_rank_total))) / K.log(2.0)
        loss_rank = tf.reduce_sum(logistic_loss, axis=1) - K.ones(batch_s, dtype='float32')
        
        return tf.reduce_mean(loss_rank)

    return loss

# Naive implementation
def loss_sca(score_vector, alpha_value=10, nb_class=256):

    # Rank loss function
    def ranking_loss_sca(y_true, y_pred):

        alpha = K.constant(alpha_value, dtype='float32')

        # Batch_size initialization
        y_true_int = K.cast(y_true, dtype='int32')
        batch_s = K.cast(K.shape(y_true_int)[0],dtype='int32')

        # Indexing the training set (range_value = (?,))
        range_value = K.arange(0, batch_s, dtype='int64')

        # Get rank and scores associated with the secret key (rank_sk = (?,))
        values_topk_logits, indices_topk_logits = tf.nn.top_k(score_vector, k=nb_class, sorted=True) # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
        rank_sk = tf.where(tf.equal(K.cast(indices_topk_logits, dtype='int64'), tf.reshape(K.argmax(y_true_int), [tf.shape(K.argmax(y_true_int))[0], 1])))[:,1] + 1 # Index of the correct output among all the hypotheses (shape(?,))
        score_sk = tf.gather_nd(values_topk_logits, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), tf.reshape(rank_sk-1, [tf.shape(rank_sk)[0], 1])])) # Score of the secret key (shape(?,))

        # Ranking Loss Initialization
        loss_rank = 0

        for i in range(nb_class):

            # Score for each key hypothesis (s_i_shape=(?,))
            s_i = tf.gather_nd(values_topk_logits, K.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), i*tf.ones([tf.shape(values_topk_logits)[0], 1], dtype='int64')]))

            # Indicator function identifying when (i == secret key)
            indicator_function = tf.ones(batch_s) - (K.cast(K.equal(rank_sk-1, i), dtype='float32') * tf.ones(batch_s))

            # Logistic loss computation
            logistic_loss = K.log(1 + K.exp(- alpha * (score_sk - s_i)))/K.log(2.0)

            # Ranking Loss computation
            loss_rank = tf.reduce_sum((indicator_function * logistic_loss))+loss_rank

        return loss_rank/(K.cast(batch_s, dtype='float32'))

    #Return the ranking loss function
    return ranking_loss_sca

### CNN network
def cnn_architecture(input_size=700,learning_rate=0.00001, alpha_value=10.0, classes=256, rkl_loss=True):
        
    # Personal design
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape, dtype='float32')

    # 1st convolutional block
    x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    
    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    
    # Logits layer
    score_layer = Dense(classes, activation=None, name='score')(x)
    predictions = Activation('softmax')(score_layer)

    # Create model
    inputs = img_input
    model = Model(inputs, predictions, name='ascad')
    optimizer = Adam(lr=learning_rate)

    if(rkl_loss==True):
        model.compile(loss=ranking_loss_optimized(score_layer, alpha_value=alpha_value),optimizer=optimizer, metrics=['accuracy'])
    
    else:
        model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=['accuracy'])

    return model


#### ASCAD helper to load profiling and attack data (traces and labels) (source : https://github.com/ANSSI-FR/ASCAD)
# Loads the profiling and attack datasets from the ASCAD database
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file  = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata']['plaintext'], in_file['Attack_traces/metadata']['plaintext'])


#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, max_lr=1e-3):
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # One Cycle Policy
    lr_manager = OneCycleLR(max_lr=max_lr, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None,verbose=True)

    callbacks=[save_model, lr_manager] 
           
    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=256)), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)
    return history


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

# Our folders
root = "./"
ASCAD_data_folder = root+"ASCAD_dataset/"
ASCAD_trained_models_folder = root+"ASCAD_trained_models/"
history_folder = root+"training_history/"
predictions_folder = root+"model_predictions/"

# Choose the name of the model
nb_epochs = 50
batch_size = 50
input_size = 700
learning_rate = 5e-3
nb_prof_traces = 45000
alpha = 0.5
nb_traces_attacks = 500
nb_attacks = 100
real_key = np.load(ASCAD_data_folder + "key.npy")

# Load the profiling traces
(X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(ASCAD_data_folder + "ASCAD.h5", load_metadata=True)

# Shuffle data
(X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

X_profiling = X_profiling.astype('float32')
X_attack = X_attack.astype('float32')

#Traces Scaling (between 0 and 1)
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X_profiling = scaler.fit_transform(X_profiling)
X_attack = scaler.transform(X_attack)

X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

# Log Initialization
sys.stdout = Logger(root+"logs/ASCAD_nbprof"+str(nb_prof_traces))

counter = 0

while (counter<2):

    #################################################
    #################################################

    ####                Training               ######

    #################################################
    #################################################

    # Choose your model
    if(counter==0):
        model = cnn_architecture(input_size=input_size, learning_rate=learning_rate, rkl_loss=False)
        model_name="ASCAD_cce_nbprof"+str(nb_prof_traces)
    else:
        model = cnn_architecture(input_size=input_size, learning_rate=learning_rate, alpha_value=alpha, rkl_loss=True)
        model_name="ASCAD_rkl_nbprof"+str(nb_prof_traces)+"_alpha"+str(alpha)

    print('\n Model name = '+model_name)

    print("\n############### Starting Training #################\n")

    start = time.time()

    # Record the metrics
    history = train_model(X_profiling[:nb_prof_traces], Y_profiling[:nb_prof_traces], X_profiling[45000:], Y_profiling[45000:], model, ASCAD_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size, max_lr=learning_rate)

    end = time.time()
    print('Training Time = %d'%(end-start))

    print("\n############### Training Done #################\n")

    # Save the metrics
    with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


    #################################################
    #################################################

    ####               Prediction              ######

    #################################################
    #################################################

    print("\n############### Starting Predictions #################\n")

    predictions = model.predict(X_attack)

    print("\n############### Predictions Done #################\n")

    np.save(predictions_folder + 'predictions_' + model_name +'.npy', predictions)


    #################################################
    #################################################

    ####            Perform attacks            ######

    #################################################
    #################################################

    print("\n############### Starting Attack on Test Set #################\n")

    avg_rank = perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=2, filename=model_name)

    print("\n Nt_GE = ")
    print(np.where(avg_rank==0))
    print("\n")

    print("\n############### Attack on Test Set Done #################\n")

    K.clear_session()
    del model

    counter+=1