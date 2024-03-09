import os
from os.path import join
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import activations
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, models
from bpmll import bp_mll_loss

from keras.utils.layer_utils import count_params

class EMGConv2DResNetModel:
    """Implementation of the 2D CNN multilabel network for EMG classification based on the article
    Olsson, Alexander E., et al. "Extraction of multi-labelled movement information from the raw HD-sEMG image with time-domain depth." Scientific reports 9.1 (2019): 1-10.
    """
    def __init__(self, config_model):
        """Class constructor to initialize the object

        Args:
            config_model (dict[str, any]): dictionary with the initialization values of the model
        """
        self.kernelDepth = 3
        self.checkpointFilepath = config_model['checkpoint_path']
        self.networkName = config_model['network_name']
        self.imgHeight = config_model['img_height']
        self.imgWidth = config_model['img_width']
        self.numFeatures = config_model['num_features']
        self.numClasses = config_model['num_classes']
        self.earlyStoppingPatient = config_model['early_stopping_patient']
        self.epochs = config_model['epochs']
        self.loss = config_model['loss']
        self.filterDivisor = 1
        self.lr = 0.01
        self.modelPath = ""
        self.optimizer = 'adam'
        self.classWeights = None
        self.outputMode = 'multilabel'
        self.print_model = config_model['print_model']
        
        if 'filter_divisor' in config_model.keys():
            self.filterDivisor = config_model['filter_divisor']

        if 'model_path' in config_model.keys():
            self.modelPath = config_model['model_path']            

        if 'lr' in config_model.keys():
            self.lr = config_model['lr']
          
        if 'optimizer' in config_model.keys():
            self.optimizer = config_model['optimizer']
            
            if 'adamW' in config_model['optimizer']:
                self.optimizer = tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=2e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        else:
            self.optimizer = tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=2e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        
        if 'class_weights' in config_model.keys():
            self.classWeights = config_model['class_weights']       
        
        if 'model_output_mode' in config_model.keys():
            self.outputMode = config_model['model_output_mode']  

    def build_model(self):
        """Build a new network model

        Returns:
            Sequential: the built model
        """
        # cnn architecture
        input = layers.Input(shape=(self.imgWidth, self.imgHeight, self.numFeatures))

        # layer 1
        # x = BatchNormalization(axis=-1, input_shape=(self.windowSize, self.numChannels)))
        x = layers.Conv2D(int(128 / self.filterDivisor), (3, 3), padding="same")(input)
        x = BatchNormalization()(x)
        x = layers.Activation(activations.relu)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=-0.01)(x)

        # layer 2
        x = layers.Conv2D(int(64 / self.filterDivisor), (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        layer_2_out = layers.Activation(activations.relu)(x)
        # layer_2_out = tf.keras.layers.LeakyReLU(alpha=-0.01)(x)

        # layer 3
        x = layers.Conv2D(int(64 / self.filterDivisor), (1, 1))(layer_2_out)
        x = BatchNormalization()(x)
        layer_3_out = layers.Activation(activations.relu)(x)
        # layer_3_out = tf.keras.layers.LeakyReLU(alpha=-0.01)(x)

        # residual 
        res_1 = layers.Add(name='res_1')([layer_2_out, layer_3_out])
        res_1 = BatchNormalization()(res_1)
        res_1 = layers.Activation(activations.relu)(res_1)
        # res_1 = tf.keras.layers.LeakyReLU(alpha=-0.01)(res_1)

        # layer 4
        x = layers.Conv2D(int(64 / self.filterDivisor), (1, 1))(res_1)
        x = BatchNormalization()(x)
        layer_4_out = layers.Activation(activations.relu)(x)
        # layer_4_out = tf.keras.layers.LeakyReLU(alpha=-0.01)(x)

        #residual
        res_2 = layers.Add(name='res_2')([res_1, layer_4_out])
        res_2 = BatchNormalization()(res_2)
        res_2 = layers.Activation(activations.relu)(res_2)
        # res_2 = tf.keras.layers.LeakyReLU(alpha=-0.01)(res_2)

        # layer 5
        x = Dropout(0.5)(res_2)
        x = layers.Flatten()(x)
        x = layers.Dense(int(512 / self.filterDivisor))(x)
        x = BatchNormalization()(x)
        x = layers.Activation(activations.relu)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=-0.01)(x)
        x = Dropout(0.5)(x)

        # layer 6
        x = layers.Dense(int(512 / self.filterDivisor))(x)
        x = BatchNormalization()(x)
        x = layers.Activation(activations.relu)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=-0.01)(x)
        x = Dropout(0.5)(x)

        # layer 7
        x = layers.Dense(int(128 / self.filterDivisor))(x)
        x = BatchNormalization()(x)
        x = layers.Activation(activations.relu)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=-0.01)(x)

        # layer 8
        x = layers.Dense(self.numClasses)(x)
        
        if(self.outputMode == 'multilabel'):
            x = layers.Activation(activations.sigmoid)(x)
            
        elif(self.outputMode == 'multiclass'):
            if not self.loss =='SparceCatCE':
                x = layers.Activation(activations.softmax)(x)

        model = models.Model(input, x)

        return model

    def compile_model(self, model):
        """Compile the model

        Args:
            model (keras model): The model to be compiled
        """
        if(self.outputMode == 'multilabel'):
            if(self.loss == 'bce'):
                model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['categorical_accuracy'])
            elif(self.loss == 'bpmll'):
                model.compile(optimizer=self.optimizer, loss=bp_mll_loss, metrics=['categorical_accuracy'])
            elif(self.loss == 'focal'):
                loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2)
                model.compile(optimizer=self.optimizer, loss=loss, metrics=['categorical_accuracy'])
            else:
                raise Exception('Loss function' + self.loss + 'not found.')
        
        elif(self.outputMode == 'multiclass'):
            if(self.loss == 'SparceCatCE'):
                print('model compiled')
                model.compile(optimizer=self.optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
            else:
                model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def load_checkpoint_model(self):
        """Load the model from file

        Returns:
            _type_: the loaded model
        """
        model = tf.keras.Model.load_model(self.modelPath)
        return model

    def load_model(self):
        """Load the model from file or build a new model

        Returns:
            _type_: the network model
        """
        print('path', os.path.isfile(self.modelPath))

        model = None
        if self.modelPath != "":
            if os.path.isfile(self.modelPath):
                model = self.load_checkpoint_model(self.modelPath)   
            else:
                print("Unable to open file", self.modelPath, '. Creating a new model.')
        
        if model is None:
            model = self.build_model()

        self.compile_model(model)      
        
        if self.print_model== True:
            model.summary()
        
        self.model = model

    def get_callbacks(self):
        """Create and retrieve the keras callbacks to be used during the training

        Returns:
            list[keras.callback]: the list with the created callbacks
        """

        if not os.path.exists(self.checkpointFilepath):
            os.mkdir(self.checkpointFilepath)
        
        if not os.path.exists(join(self.checkpointFilepath, self.networkName)):
            os.mkdir(join(self.checkpointFilepath, self.networkName))
        
        log_dir = join(self.checkpointFilepath, "logs/fit/", self.networkName)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=join(self.checkpointFilepath, self.networkName), monitor='val_loss', mode='min', save_best_only=True)
        model_earlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.earlyStoppingPatient, verbose=0, mode='min', baseline=None, restore_best_weights=False)
        csvLogger = tf.keras.callbacks.CSVLogger(join(self.checkpointFilepath, self.networkName, self.networkName + '_history.csv'), separator=',',append=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print(join(self.checkpointFilepath, self.networkName))
        return [model_earlyStopping_callback, model_checkpoint_callback, csvLogger, tensorboard_callback]

    def train(self, train_dataset, val_dataset):
        verbose=2
        if self.print_model == False:
            verbose = 0

        if self.classWeights is None:
            print('no weights')
            history = self.model.fit(train_dataset, epochs=self.epochs, callbacks=self.get_callbacks(), validation_data=val_dataset,verbose=verbose)
        else:
            print('with weights')
            history = self.model.fit(train_dataset, epochs=self.epochs, callbacks=self.get_callbacks(), validation_data=val_dataset, class_weight=self.classWeights,verbose=verbose)

        return history
    
    
    def count_params(self):

        trainable_count =  count_params(self.model.trainable_weights)
        non_trainable_count = count_params(self.model.non_trainable_weights)
        
        return trainable_count, non_trainable_count
