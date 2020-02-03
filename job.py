from __future__ import absolute_import, division, print_function, unicode_literals

def main_fun(args, ctx):
	import numpy as np
	import tensorflow as tf
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, DenseFeatures
	from tensorflow import feature_column
	import tensorflow.keras as keras
	from tensorflowonspark import compat, TFNode

	strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

	def buildAndCompileModel():
		# Initiating model
		model = Sequential()

		# Building model structure
		model.add(Input(shape=(1025, 50, 1)))
		model.add(Conv2D(64, kernel_size=[5,5], padding='same', activation='relu', data_format='channels_last'))
		model.add(MaxPool2D(pool_size=[5,5], data_format='channels_last'))
		model.add(Flatten())
		model.add(Dropout(0.2))
		model.add(Dense(32))
		# model.add(Dense(64, activation='relu'))
		model.add(Dense(1, activation='sigmoid'))

		# Compiling model
		model.compile(
			loss='binary_crossentropy',
			optimizer='adam',
			metrics=['accuracy']
		)

		return model



	tfFeed = TFNode.DataFeed(ctx.mgr, False)

	def rddGenerator():
		while not tfFeed.should_stop():
			batch = tfFeed.next_batch(1)
			if len(batch) > 0:
				example = batch[0]

				# Splitting into X and y
				X = np.array(example[1]).astype(np.float32)
				y = np.array(example[0])

				# Encoding labels
				_, y = np.unique(y, return_inverse=True)
				y = y.astype(np.float32)

				# Adjusting data shape
				X = X.reshape(-1, 50, 1)

				# Shuffling X and y in unison
				# from sklearn.utils import shuffle
				# X, y = shuffle(X, y, random_state = 0)

        # image = np.array(example[0]).astype(np.float32) / 255.0
        # image = np.reshape(image, (28, 28, 1))
        # label = np.array(example[1]).astype(np.float32)
        # label = np.reshape(label, (1,))

				yield (X, y)
			else:
				return

	# Creating Tensorflow Dataset
	ds = tf.data.Dataset.from_generator(rddGenerator, (tf.float32, tf.float32), (tf.TensorShape([1025, 50, 1]), tf.TensorShape([1])))
	ds = ds.batch(args.batch_size)

	# Instantiating Model
	with strategy.scope():
		multiWorkerModel = buildAndCompileModel()

	# Defining Training Parameters
	stepsPerEpoch = 600 / 1
	stepsPerWorker = stepsPerEpoch / 1
	maxStepsPerWorker = stepsPerWorker * 0.9

	# Fitting Model
	multiWorkerModel.fit(x = ds, epochs = 2, steps_per_epoch = stepsPerWorker)

	from tensorflow_estimator.python.estimator.export import export_lib
	exportDir = export_lib.get_timestamped_export_dir(args.export_dir)
	compat.export_saved_model(multiWorkerModel, exportDir, ctx.job_name == 'chief')

  # terminating feed tells spark to skip processing further partitions
	tfFeed.terminate()


if __name__ == '__main__':
	# Initializing Spark
	import findspark
	findspark.init()

	# Importing Spark-related Things
	import argparse
	from pyspark import SparkContext
	from pyspark.sql import SparkSession

	# Importing TensorflowOnSpark
	from tensorflowonspark import TFCluster

	# Importing other packages
	import numpy as np
	import io
	import os
	import librosa
	import soundfile as sf

	# Creating Spark Session
	spark = SparkSession.builder \
    .master('local') \
    .appName('AudioAttempt') \
    .config('spark.executor.memory', '8gb') \
    .getOrCreate()

	# Creating Spark context
	sc = spark.sparkContext

	# Still not entirely sure what this is
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
	parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=1)
	parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
	parser.add_argument("--images_labels", help="path to MNIST images and labels in parallelized format")
	parser.add_argument("--model_dir", help="path to save checkpoint", default="mnist_model")
	parser.add_argument("--export_dir", help="path to export saved_model", default="mnist_export")
	parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

	args = parser.parse_args()
	print("args:", args)


	# - - - - -

	# Function to carve out class from file path
	def carveClassName(x):
		return x.split('/')[-2]

	# Function to convert audio stored as binary in-memory to numerical data
	def binaryToNumerical(x):
		return sf.read(io.BytesIO(x))[0]

	# Function to perform fourier transformation on audio snippets to get energy in different frequency ranges
	def fourierTransformation(x):
		audio = librosa.amplitude_to_db(abs(librosa.stft(x, hop_length=321)))

		# Padding audio up to one second length if it is shorter
		if audio.shape[1] < 50:
			filler = np.zeros((1025, 50 - audio.shape[1]))
			audio = np.concatenate((audio, filler), axis = 1)
				
		return audio

	# - - - - -

	# Loading audio data into cluster memory as binary data
	paths = 'data/local/bird,data/local/bed,data/local/cat'

	# Loading data into memory
	baseAudio = sc.binaryFiles(paths)

	# Applying Transformations To Data
	convertedAndLabeledAudio = baseAudio.map(lambda x: [carveClassName(x[0]), binaryToNumerical(x[1])])
	transformedAudio = convertedAndLabeledAudio.map(lambda x: [x[0], fourierTransformation(x[1])])

	# Defining Cluster
	cluster = TFCluster.run(sc, main_fun, args, args.cluster_size, num_ps = 0, tensorboard = args.tensorboard, input_mode=TFCluster.InputMode.SPARK, master_node='chief')

	cluster.train(transformedAudio, args.epochs)
	cluster.shutdown()



	# Fix end of sequence thing (Only 30 samples wtf?)

