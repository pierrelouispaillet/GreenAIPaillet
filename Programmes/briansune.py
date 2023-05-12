import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
import cv2

# Python3 compatibility code
import sys
if sys.version_info >= (3, 0):
	from functools import reduce

class FFTConvTest:

	def __init__(self, operations, initialization=None, learning_rate=0.0002, spectral_regularization_alpha=2.0):
		# Load MNIST data
		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9, allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

		# Setup inputs into the network for images and targets
		self.image_vector = tf.placeholder(tf.float32, shape=[None, 784])
		self.one_hot_class = tf.placeholder(tf.float32, shape=[None, 10])

		# Reshape MNIST vectors into images, and center
		self.image_matrix = tf.reshape(self.image_vector, shape=[-1, 28, 28, 1]) - .5

		# Switch between a network with FFT convolutions
		conv_op = self.conv2d
		if operations == 'fft':
			conv_op = self.fft_conv
		elif operations == 'fft_pure':
			conv_op = self.fft_conv_pure

		# Handle forced initialization
		self.initialization = initialization if initialization is not None else {}

		# Build network
		conv1, self.spatial_conv1, self.spectral_conv1 = conv_op(self.image_matrix, filters=15, width=10, height=10,
																 stride=1, name='conv1')
		pool1 = self.maxpool2d(conv1, 2, 2)
		conv2, self.spatial_conv2, self.spectral_conv2 = conv_op(pool1, filters=1, width=6, height=6,
																 stride=1, name='conv2')
		pool2 = self.maxpool2d(conv2, 2, 2)
		fc1 = self.linear(self.flatten(pool2), 1024, name='fft_fc1')
		output = self.linear(fc1, 10, activation='none', name='fft_output')

		print("Model Free Parameters: {}".format(self.variables()))

		# Build cost and optimizer
		self.output_softmax = tf.nn.softmax(output)
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.one_hot_class * tf.log(self.output_softmax), reduction_indices=[1]))
		regularization = tf.reduce_mean(tf.abs(self.spectral_conv1)) + tf.reduce_mean(tf.abs(self.spectral_conv2))
		self.error = cross_entropy + spectral_regularization_alpha * regularization
		optimizer = tf.train.AdamOptimizer(learning_rate)
		self.train_step = optimizer.minimize(self.error)

		self.sess.run(tf.initialize_all_variables())

		# Initialize visualization window
		self.window_name = "preview-operations=" + operations
		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
		cv2.startWindowThread()

	def conv2d(self, source, filters, width, height, stride, activation='relu', name='conv2d'):
		# Normal convolution layer
		in_channels = source.get_shape().as_list()[3]

		with tf.variable_scope(name):
			spatial_filter = tf.get_variable("weight", [height, width, in_channels, filters],
								initializer=tf.truncated_normal_initializer(0, stddev=0.01), dtype=tf.float32)
			b = tf.Variable(tf.constant(0.1, shape=[filters]), name="bias")

			# Run the filter through ifft(fft(x)) to demonstrate that those functions are inverses of one another
			spatial_filter_for_fft = tf.transpose(spatial_filter, [2, 3, 0, 1])

			# Compute the spectral filter for visualization
			spectral_filter = tf.fft2d(tf.complex(spatial_filter_for_fft, spatial_filter_for_fft * 0.0))

		conv = tf.nn.conv2d(source, spatial_filter, strides=[1, stride, stride, 1], padding='SAME')
		output = tf.nn.bias_add(conv, b)
		output = tf.nn.relu(output) if activation is 'relu' else output

		return output, spatial_filter, spectral_filter

	def linear(self, source, neurons, activation='relu', name='linear'):
		# Normal fully connected layer
		in_channels = source.get_shape().as_list()[1]

		with tf.variable_scope(name):
			w = tf.Variable(tf.truncated_normal(shape=[in_channels, neurons], mean=0, stddev=0.1))
			b = tf.Variable(tf.constant(0.1, shape=[neurons]), name="bias")

		output = tf.nn.bias_add(tf.matmul(source, w), b)
		output = tf.nn.relu(output) if activation is 'relu' else output

		return output

	def flatten(self, source):
		# Converts a tensor into a vector
		return tf.reshape(source, [-1, np.prod(source.get_shape().as_list()[1:])])

	def maxpool2d(self, source, width, stride):
		# Applies maxpool operation to a tensor
		return tf.nn.max_pool(source, ksize=[1, width, width, 1], strides=[1, stride, stride, 1], padding='SAME')

	def variables(self):
		# Count up free parameters (variables) in the network
		return sum([sum([reduce(lambda x, y: x * y, l.get_shape().as_list() or [1]) for l in e]) for e in
					[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]])

	def fft_conv(self, source, filters, width, height, stride, activation='relu', name='fft_conv'):
		# This function implements a convolution using a spectrally parameterized filter with the normal
		# tf.nn.conv2d() convolution operation. This is done by transforming the spectral filter to spatial
		# via tf.ifft2d()

		channels = source.get_shape().as_list()[3]

		with tf.variable_scope(name):
			init = self.random_spatial_to_spectral(channels, filters, height, width)

			if name in self.initialization:
				init = self.initialization[name]

			# Option 1: Over-Parameterize fully in the spectral domain
			w_real = tf.Variable(init.real, dtype=tf.float32, name='real')
			w_imag = tf.Variable(init.imag, dtype=tf.float32, name='imag')
			w = tf.cast(tf.complex(w_real, w_imag), tf.complex64)

			# Option 2: Parameterize only 'free' parameters in the spectral domain to enforce conjugate symmetry
			#		   This is very slow.
			#w = self.spectral_to_variable(init)

			b = tf.Variable(tf.constant(0.1, shape=[filters]))

		# Transform the spectral parameters into a spatial filter
		# and reshape for tf.nn.conv2d
		complex_spatial_filter = tf.ifft2d(w)
		spatial_filter = tf.real(complex_spatial_filter)
		spatial_filter = tf.transpose(spatial_filter, [2, 3, 0, 1])

		conv = tf.nn.conv2d(source, spatial_filter, strides=[1, stride, stride, 1], padding='SAME')
		output = tf.nn.bias_add(conv, b)
		output = tf.nn.relu(output) if activation is 'relu' else output

		return output, spatial_filter, w

	def batch_fftshift2d(self, tensor):
		# Shifts high frequency elements into the center of the filter
		indexes = len(tensor.get_shape()) - 1
		top, bottom = tf.split(tensor, 2, indexes - 1)
		tensor = tf.concat([bottom, top], indexes - 1)
		left, right = tf.split(tensor, 2, indexes)
		tensor = tf.concat([right, left], indexes)
		
		return tensor

	def batch_ifftshift2d(self, tensor):
		# Shifts high frequency elements into the center of the filter
		indexes = len(tensor.get_shape()) - 1
		left, right = tf.split(tensor, 2, indexes)
		tensor = tf.concat([right, left], indexes)
		top, bottom = tf.split(tensor, 2, indexes - 1)
		tensor = tf.concat([bottom, top], indexes - 1)
		
		return tensor

	def fft_conv_pure(self, source, filters, width, height, stride, activation='relu', name='fft_conv'):
		# This function applies the convolutional filter, which is stored in the spectral domain, as a element-wise
		# multiplication between the filter and the image (which has been transformed to the spectral domain)

		_, input_height, input_width, channels = source.get_shape().as_list()

		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			init = self.random_spatial_to_spectral(channels, filters, height, width)

			if name in self.initialization:
				init = self.initialization[name]

			# Option 1: Over-Parameterize fully in the spectral domain
			w_real = tf.Variable(init.real, dtype=tf.float32, name='real')
			w_imag = tf.Variable(init.imag, dtype=tf.float32, name='imag')
			w = tf.cast(tf.complex(w_real, w_imag), tf.complex64)

			# Option 2: Parameterize only 'free' parameters in the spectral domain to enforce conjugate symmetry
			#		   This is very slow.
			#w = self.spectral_to_variable(init)

			# Option 3: Parameterize in the spatial domain
			#w = tf.get_variable(
			#	"weight_fft",		# this name MUST MUST MUST be unique!!!!
			#	[channels, filters, height, width],
			#	initializer=tf.truncated_normal_initializer(0, stddev=0.01),
			#	dtype=tf.float32
			#)
			#w = tf.fft2d(tf.complex(w, w*0.0))

			b = tf.Variable(tf.constant(0.1, shape=[filters]))

		# Add batch as a dimension for later broadcasting
		w = tf.expand_dims(w, 0)  # batch, channels, filters, height, width

		# Prepare the source tensor for FFT
		source = tf.transpose(source, [0, 3, 1, 2])  # batch, channel, height, width
		source_fft = tf.fft2d(tf.complex(source, 0.0 * source))

		# Prepare the FFTd input tensor for element-wise multiplication with filter
		source_fft = tf.expand_dims(source_fft, 2)  # batch, channels, filters, height, width
		source_fft = tf.tile(source_fft, [1, 1, filters, 1, 1])

		# Shift, then pad the filter for element-wise multiplication, then unshift
		w_shifted = self.batch_fftshift2d(w)
		height_pad = (input_height - height) // 2
		width_pad = (input_width - width) // 2
		
		# Pads with zeros
		w_padded = tf.pad(
			w_shifted,
			[[0, 0], [0, 0], [0, 0], [height_pad, height_pad], [width_pad, width_pad]],
			mode='CONSTANT'
		)
		w_padded = self.batch_ifftshift2d(w_padded)

		# Convolve with the filter in spectral domain
		conv = source_fft * tf.conj(w_padded)

		# Sum out the channel dimension, and prepare for bias_add
		# Note: The decision to sum out the channel dimension seems intuitive, but
		#	   not necessarily theoretically sound.
		conv = tf.real(tf.ifft2d(conv))
		conv = tf.reduce_sum(conv, reduction_indices=1)  # batch, filters, height, width
		conv = tf.transpose(conv, [0, 2, 3, 1])  # batch, height, width, filters

		# Drop the batch dimension to keep things consistent with the other conv_op functions
		w = tf.squeeze(w, [0])  # channels, filters, height, width

		# Compute a spatial encoding of the filter for visualization
		spatial_filter = tf.ifft2d(w)
		spatial_filter = tf.transpose(spatial_filter, [2, 3, 0, 1])  # height, width, channels, filters

		# Add the bias (in the spatial domain)
		output = tf.nn.bias_add(conv, b)
		output = tf.nn.relu(output) if activation is 'relu' else output

		return output, spatial_filter, w

	def is_valid_spectral(self, tensor):
		# Check a tensor for conjugate symmetry and real-valuedness
		tensor = np.fft.fftshift(tensor, [2, 3])

		filters, channels, height, width = tensor.shape
		for f in range(filters):
			for c in range(channels):
				for h in range(height):
					for w in range(width):
						p = tensor[f, c, (height - h) % height, (width - w) % width]
						if np.abs(tensor[f, c, h, w] - np.conjugate(p)) > 0.0:
							return False, np.abs(tensor[f, c, h, w] - np.conjugate(p)), (h, w)

		return True

	def random_spatial_to_spectral(self, channels, filters, height, width):
		# Create a truncated random image, then compute the FFT of that image and return it's values
		# used to initialize spectrally parameterized filters
		# an alternative to this is to initialize directly in the spectral domain
		w = tf.truncated_normal([channels, filters, height, width], mean=0, stddev=0.01)
		fft = tf.fft2d(tf.complex(w, 0.0 * w), name='spectral_initializer')
		return fft.eval(session=self.sess)

	def spectral_to_variable(self, init):
		# Parameterize, via a matrix of single variables, only free parameters
		# this is extremely slow, but confirms that steps in the direction of
		# the gradient always result in valid spectral forms of valid real images

		channels, filters, height, width = init.shape

		hheight = height // 2
		hwidth = width // 2

		assert height % 2 == 0 and width % 2 == 0, 'filter size must be even (for now)'

		init = np.fft.fftshift(init, [2, 3])

		reals = init.real.tolist()
		imags = init.imag.tolist()

		for c in range(channels):
			for f in range(filters):
				I = [(0, 0), (hheight, 0), (0, hwidth), (hheight, hwidth)]

				for h, w in I:
					reals[c][f][h][w] = tf.Variable(init[c, f, h, w].real)
					imags[c][f][h][w] = 0.0

				for h in range(height):
					for w in range(width):

						if (h, w) not in I:
							imags[c][f][h][w] = tf.Variable(init[c, f, h, w].imag)
							reals[c][f][h][w] = tf.Variable(init[c, f, h, w].real)
							reals[c][f][(height - h) % height][(width - w) % width] = reals[c][f][h][w]
							imags[c][f][(height - h) % height][(width - w) % width] = -imags[c][f][h][w]

							I += [(h, w), ((height - h) % height, (width - w) % width)]


		return self.batch_ifftshift2d(tf.complex(tf.convert_to_tensor(reals), tf.convert_to_tensor(imags)))

	def visualize_filters(self, spectral, spatial):
		# Flattens a weight matrix into a 2D image representation with channels and filters
		# unrolled, then combines spectral and spatial 2D representations vertically

		# Get the spectral values, shift the high frequency into the center
		# and reshape to a common format
		spectral_filters = spectral.eval(session=self.sess)

		spectral_filters_shifted = np.fft.fftshift(spectral_filters, axes=[3, 2])
		spectral_filters_shifted = np.transpose(spectral_filters_shifted, [1, 0, 2, 3])

		# Get the spatial values and reshape to a common format
		spatial_filters = spatial.eval(session=self.sess)
		spatial_filters = np.transpose(spatial_filters, [3, 2, 0, 1])

		def unroll_filters(filters):
			filters = np.transpose(filters, [3, 2, 0, 1]).real
			filters = np.hstack(np.split(filters, filters.shape[2], axis=2))
			filters = np.vstack(np.split(filters, filters.shape[3], axis=3))[:, :, 0, 0]
			filters -= np.min(filters)
			filters /= np.max(filters)
			return filters

		spectral_filters_shifted = unroll_filters(spectral_filters_shifted)
		spatial_filters = unroll_filters(spatial_filters)

		cv2.imshow(self.window_name, np.vstack((spectral_filters_shifted, spatial_filters)))
	
	def accuracy(self):
		correct_predictions = tf.equal(tf.argmax(self.output_softmax, 1), tf.argmax(self.one_hot_class, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
		return accuracy.eval(session=self.sess, feed_dict={self.image_vector: self.mnist.test.images[:1000],
														   self.one_hot_class: self.mnist.test.labels[:1000]})

	def train(self):
		for _ in tqdm(range(1000)):
			images, labels = self.mnist.train.next_batch(50)

			self.visualize_filters(self.spectral_conv1, self.spatial_conv1)

			_, error = self.sess.run([self.train_step, self.error], feed_dict={self.image_vector: images,
																			   self.one_hot_class: labels})
		
		return self.accuracy()


if __name__ == "__main__":
	baseline = FFTConvTest(operations='conv', spectral_regularization_alpha=1)
	print("Baseline Accuracy: {}".format(baseline.train()))

	fft = FFTConvTest(operations='fft', spectral_regularization_alpha=2.3)
	print("FFT Accuracy: {}".format(fft.train()))

	fftpure = FFTConvTest(operations='fft_pure', spectral_regularization_alpha=1)
	print("FFTPure Accuracy: {}".format(fftpure.train()))
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()