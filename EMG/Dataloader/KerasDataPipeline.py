import tensorflow as tf

class KerasDataPipeline:
    """_summary_
    """
    def __init__(self, dataset_generator, input_shape, output_shape):
        """Class constructor to initialize the object

        Args:
            dataset_generator (_type_): dataset generator
            input_shape (tuple or list): the input dimension of the model
            output_shape (tuple or list): the output dimension of the model
        """
        self.datasetGenerator = dataset_generator
        self.inputShape = input_shape
        self.outputShape = output_shape
        
        self.datasetPipeline = tf.data.Dataset.range(len(self.datasetGenerator))
        self.datasetPipeline = self.datasetPipeline.shuffle(buffer_size=len(self.datasetGenerator), seed=0, reshuffle_each_iteration=True)
        self.datasetPipeline = self.datasetPipeline.map(lambda i: tf.py_function(func=self.getDataItem, inp=[i], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
        self.datasetPipeline = self.datasetPipeline.map(self.fixupShape, num_parallel_calls=tf.data.AUTOTUNE)
        self.datasetPipeline = self.datasetPipeline.prefetch(tf.data.AUTOTUNE)

    def getDataItem(self, i):
        """Retrieve the i-th batch from the 

        Args:
            i (int): The index of the batch

        Returns:
            typle[keras tensor]: The x- and y-data of the batch
        """
        i = i.numpy() # Decoding from the EagerTensor object
        x, y = self.datasetGenerator.__getitem__(i)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        return x, y
    
    def fixupShape(self, x, y):
        """Fix shape of the tensors

        Args:
            x (keras tensor): Tensor with x_data
            y (keras tensor): Tensor with y_data

        Returns:
            tuple[keras tensor]: Fixed tensors
        """
        x.set_shape(self.inputShape) 
        y.set_shape(self.outputShape)
        return x, y