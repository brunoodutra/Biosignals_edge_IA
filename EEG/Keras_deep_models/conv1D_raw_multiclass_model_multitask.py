import os

import tensorflow as tf

from keras.utils.np_utils import to_categorical

import json

from keras import backend as K
from tensorflow.keras import layers, activations, models
from tensorflow.keras.layers import BatchNormalization, MaxPooling1D, Dropout

def weighted_categorical_crossentropy(weights):
    """
    from https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

class CONV1DRAW:
    """1D CNN model for classification
    """
    def __init__(self, name= "conv1d-raw", model_path= None, checkpoint_filepath= "CONV1D/model_conv1d_eeg/m0", factor=1, kernLength = 256, D=1, params=None, print_model=False):
        """Class constructor to initialize the object

        Args:
            config_model (dict[str, any]): dictionary with the initialization values of the model
        """
       
        self.kernelDepth = 3
        
        self.checkpointFilepath = checkpoint_filepath
        self.networkName = name
        self.numClasses = params['num_classes']
        self.numChannels =  params['num_channels']
        self.window_size =  params['window_size']
        self.numSamples =  params['num_samples']
        self.numHeights = params['num_heights']
        self.epochs = params['epochs']
       
        self.kernLength = kernLength
        self.dropoutRate = params['dropout_rate']
        
        self.neurons_conv=params['neurons_conv']
        self.neurons_dense=params['neurons_dense']

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        if model_path != None:
            self.modelPath = model_path
        else:
            self.modelPath = ""
            
        self.print_model = print_model
  

    def build_model(self):
        """Build a new network model

        Returns:
            Sequential: the built model
        """
        # cnn architecture


        input1      = tf.keras.layers.Input(shape = (self.window_size,self.numChannels))
        layers      = self.common_layers(input1)
        freq_branch = self.head_layer(layers, self.numClasses, 'freq_output')
        task_branch = self.head_layer(layers, 2, 'task_output')
        
        model = tf.keras.models.Model(inputs=input1,
                 outputs = [freq_branch, task_branch],
                 name="multitask")

        return model

        
    def common_layers(self, input1):
        """Common layers to the network model

        Returns:
            Graph: the common layers model
        """

        ##################################################################
        # CNN architecture
        conv_layer = layers.Conv1D(self.neurons_conv[0], (3), padding="same")(input1)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = layers.Activation(activations.relu)(conv_layer)
        conv_layer = MaxPooling1D(pool_size=(3), strides=1, padding="same")(conv_layer)
        conv_layer = layers.Conv1D(self.neurons_conv[1], (3), padding="same")(conv_layer)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = layers.Activation(activations.relu)(conv_layer)
        conv_layer = MaxPooling1D(pool_size=(3), strides=2, padding="same")(conv_layer)
        
        ##################################################################

        return conv_layer
    
    def head_layer(self, conv_layer,num_classes,output_name=None):
        
        head1 = layers.Conv1D(self.neurons_conv[1]/2, (3), padding="same")(conv_layer)
        head1 = BatchNormalization()(head1)
        head1 = layers.Activation(activations.relu)(head1)
        head1 = MaxPooling1D(pool_size=(3), strides=2, padding="same")(head1)

        head1 = Dropout(0.5)(head1)
        head1 = layers.Flatten()(head1)

        x = layers.Dense(self.neurons_dense[0])(head1)
        x = BatchNormalization()(x)
        x = layers.Activation(activations.relu)(x)
        x = Dropout(0.5)(x)

        x = layers.Dense(self.neurons_dense[1])(x)
        x = BatchNormalization()(x)
        x = layers.Activation(activations.relu)(x)
        x = Dropout(0.5)(x)

        x = layers.Dense(num_classes)(x)
        x = layers.Activation(activations.softmax, name=output_name)(x)

        return x

    def compile_model(self, model, custom_optimizer, weight_freq=None, weight_task=None):
        """Compile the model

        Args:
            model (_type_): The model to be compiled
        """
        
        if weight_freq is None:
            loss_freq = 'categorical_crossentropy'
        else:
            loss_freq = weighted_categorical_crossentropy(weight_freq)
            
        if weight_task is None:
            loss_task = 'categorical_crossentropy'
        else:
            loss_task = weighted_categorical_crossentropy(weight_task)
        
        # model.compile(loss='categorical_crossentropy', optimizer=custom_optimizer, metrics=['categorical_accuracy'])#
        model.compile(optimizer=custom_optimizer, 
              # loss={
              #     'freq_output': 'categorical_crossentropy' , 
              #     'task_output': 'categorical_crossentropy', 
              # },
              loss={
                  'freq_output': loss_freq, 
                  'task_output': loss_task, 
              },
               # class_weight={0: 0.67, 1: 2
                    # 'gest_output': {0: 0.67, 1: 2},
                    # 'task_output': {0: 0.67, 1: 2},
                # },
              metrics={
                  'freq_output': 'categorical_accuracy', 
                  'task_output': 'categorical_accuracy',
                  
              })        
  
        
        return model

    def load_checkpoint_model(self):
        """Load the model from file

        Returns:
            _type_: the loaded model
        """
        model = tf.keras.models.load_model(self.checkpointFilepath)
        return model

    def load_model(self):
        """Load the model from file or build a new model

        Returns:
            _type_: the network model
        """
        print('path', os.path.isfile(self.checkpointFilepath))

        model = None
        if self.modelPath != "":
            if os.path.isfile(self.checkpointFilepath):
                model = self.load_checkpoint_model(self.checkpointFilepath)   
            else:
                print("Unable to open file", self.checkpointFilepath, '. Creating a new model.')
        
        if model is None:
            model = self.build_model()

        if self.print_model:
            model.summary()
        
        
        return model

    def get_callbacks(self):
        """Create and retrieve the keras callbacks to be used during the training

        Returns:
            list[keras.callback]: the list with the created callbacks
        """
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpointFilepath, monitor='val_loss', mode='min', save_best_only=True)
        model_reduceLr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor = 0.2,patience = 5,min_lr=1e-14)
        model_earlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=False)
        csvLogger = tf.keras.callbacks.CSVLogger(self.checkpointFilepath + self.networkName + '_history.csv', separator=',',append=True)

        return [model_earlyStopping_callback, model_checkpoint_callback, model_reduceLr_callback, csvLogger]

    def train(self, model, train_dataset, val_dataset):

        if self.print_model:
            history = model.fit(train_dataset, epochs=self.epochs, callbacks=self.get_callbacks(), validation_data=val_dataset)
        else:
            history = model.fit(train_dataset, epochs=self.epochs, callbacks=self.get_callbacks(), validation_data=val_dataset, verbose=0)
           

        return history


