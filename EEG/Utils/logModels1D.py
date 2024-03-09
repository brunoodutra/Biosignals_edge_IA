#!/usr/bin/env python
# coding: utf-8

import h5py
import sys
import mlflow
import json

def log_into_mlflow_Conv1D(experiment, params_file, factor, D, trainable_count, non_trainable_count, learning_rate, opt, test_performance, test_precision, test_recall, test_f1_score, rep,train_performance,val_performance):
    
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        
        # file loads paramms 
        f = open(params_file)
        params = json.load(f)

        #
        mlflow.log_param('dataset_path', params['dataset_path'])
        mlflow.log_param('h5_file_name', params['h5_file_name'])
        mlflow.log_param('preprocessing', params['preprocessing'])

        #dataset params 
        mlflow.log_param('extracted_frequencies', params['extracted_frequencies'])
        mlflow.log_param('features_type', params['features_type'])
        mlflow.log_param('num_eeg_channels', params['num_channels'])
        mlflow.log_param('num_class', params['num_classes'])
        mlflow.log_param('samples', params['num_samples'])
        mlflow.log_param('window_size', params['window_size'])

        # model params 
        mlflow.log_param('model_name', params['model_name'])
        mlflow.log_param('epochs', params['epochs'])
        mlflow.log_param('batch_size', params['batchsize'])
        mlflow.log_param('output_function', params['output_function'])

        
        
        mlflow.log_param('trainable_count', trainable_count)
        mlflow.log_param('non_trainable_count', non_trainable_count)

        mlflow.log_param('factor', factor)
        mlflow.log_param('D', D)

        
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('optimizer', opt)

        
        #test metrics
        mlflow.log_metric('train_accuracy', train_performance[1])#history.history['categorical_accuracy'][-1])
        mlflow.log_metric('val_accuracy', val_performance[1])#history.history['val_categorical_accuracy'][-1])

        mlflow.log_metric('test_accuracy', test_performance[1])
        mlflow.log_metric('test_precision', test_precision)
        mlflow.log_metric('test_recall', test_recall)
        mlflow.log_metric('test_f1_score', test_f1_score)

        mlflow.log_metric('Repetition', rep)

        #mlflow.log_metric('CM', labels)

        #mlflow.log_artifact('my_model.h5')    
        # End the MLflow run
        #mlflow.end_run()
