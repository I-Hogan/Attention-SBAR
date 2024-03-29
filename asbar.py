# Skeleton Based Activity Recognition using Attention 
#
# Version Num: 0.0.6
#
# Isaac Hogan
# 14iach@queensu.ca

# Data conversion tool: https://github.com/shahroudy/NTURGB-D

print("\nloading libraries...")

import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
import glob
import random
import math
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
print("TensorFlow Version:", tf.__version__)

print()

# ---------- Functions ----------

def check_sample(sample):
	"""
	check if a sample always has one subject
	"""
	if not all(i == 1 for i in sample['nbodys']):
		return False
	return True
	
def scale_clips(set):
	"""
	linearly scales all clips in the set so that the largest joint coordinate is 1 and the smallest is 0
	"""
	for i in range(len(set)):
		max = np.amax(np.amax(np.amax(set[i]['skel_body0'], axis=2), axis= 1), axis= 0)
		min = np.amin(np.amin(np.amin(set[i]['skel_body0'], axis=2), axis= 1), axis= 0)
		set[i]['skel_body0'] = np.subtract(set[i]['skel_body0'], min)
		set[i]['skel_body0'] = np.divide(set[i]['skel_body0'], (max - min))
		
def add_class(set):
	"""
	add class number as `class_num` to the NTU RGB+D dictionary
	"""
	for i in range(len(set)):
		c_num = int(set[i]['file_name'][-3:])
		set[i]['class_num'] = c_num
		
def load_ntu(data_file, train_size=0.8, test_size=0.2, evaluation_mode="default", first_classes=False, single_pose=False):
	"""
	Creates a training set and a testing set from the numpy-converted NTU RGB+D Skeleton dataset

	NTU RGB+D data is a dictionary with the following keys:
	 `file_name`: file's name
	 `nbodys`: it's a list with same length of the sequence. it represents the number of the actors in each frame.
	 `njoints`: the number of the joint node in the skeleton, it's a constant here
	 `skel_bodyx`: the skeleton coordinate with the shape of `(nframe, njoint, 3)`, the x denotes the id of the acting person in each frame. (starting at 0)
	 `rgb_bodyx`: the projection of the skeleton coordinate in RGBs.
	 `depth_bodyx`: the projection of the skeleton coordinate in Depths

	added entries:
	 `class_num`: the class number, extracted from the name

	"""
	print("loading data...")
	
	train_set = []
	test_set = []

	#Load full file list
	files = [f for f in glob.glob(data_file + '*')]
	num_classes = 120 
			
	if first_classes:
		#load the first 5 classes
		files = []
		num_classes = 5
		for i in range(num_classes):
			files = files + [f for f in glob.glob(data_file + '*A00' + str(i + 1) + '*')]
	
	if evaluation_mode == "default":

		random.shuffle(files)

		train_samples = math.floor(len(files)*train_size)
		test_samples = math.floor(len(files)*test_size)

		prev_sampels = 0

		#creates the training set
		for i in range(train_samples):
			sample = np.load(files[i+prev_sampels],allow_pickle=True).item()
			if single_pose:
				if check_sample(sample):
					train_set.append(sample)
			else:
				train_set.append(sample)
		prev_sampels += train_samples

		#creates the test set
		for i in range(test_samples):
			sample = np.load(files[i+prev_sampels],allow_pickle=True).item()
			if single_pose:
				if check_sample(sample):
					test_set.append(sample)
			else:
				test_set.append(sample)
		prev_sampels += test_samples

	#elif evaluation_mode == "cv":
		

	else:
		print(evaluation_mode, "is invalid evaluation mode")
		print("options: default, cv, cs")

	add_class(train_set)
	add_class(test_set)

	if SCALE_CLIPS:
		print("scaling clips...")
		scale_clips(train_set)
		scale_clips(test_set)

	return train_set, test_set
			
def get_vals_labels(set):
	"""
	takes a set from the NTU RGB+D dictionary and returns training and testing sets and labels
	"""
	x = []
	y = []
	
	for clip in set:
		x.append(clip['skel_body0'])
		y.append(clip['class_num'])
		
	return x, y
	
def normalize_clip_length(set, new_len):
	"""
	pad or cut each clip in the set to a new length
	"""
	print("normalizing clip length...")

	for clip in range(len(set)):
		if set[clip].shape[0] > new_len:
			set[clip] = set[clip][0:new_len]
		set[clip] = np.pad(set[clip], \
			pad_width=((0,(new_len - set[clip].shape[0])), (0,0), (0,0)), \
			mode='constant', constant_values=0)
		

def matplotlib_clip_vid(clip):
	"""
	show activity clip
	"""
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation
	
	fig = plt.figure()
	
	clip_t = clip.transpose((0,2,1))
	
	def update(frame_number):
		ax = fig.add_subplot(projection='3d')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		scatter = ax.scatter(clip_t[frame_number][0], clip_t[frame_number][1], clip_t[frame_number][2])
		ax.set_xlim(0,1)
		ax.set_ylim(0,1)
		ax.set_zlim(0,1)
		return scatter
		
	ani = FuncAnimation(fig, update, interval=10)
	plt.show()

# ---------- Classes ----------

class ModifySkeletons(tf.keras.layers.Layer):
	"""
	Rotates and scales skeletons by random values
	"""
	def __init__(self):
		super(ModifySkeletons, self).__init__()
		
	def call(self, input, max_val=1., min_val=0., training=None):
	
		if training:
			theta = random.uniform(0., 360.)
			rad = theta*math.pi/180.
			scale = random.uniform(.5, 2)
		
			translate_val = (min_val - max_val) / 2.
			translate_forward = tf.constant([[1,0,0,translate_val],
										[0,1,0,translate_val],
										[0,0,1,translate_val],
										[0,0,0,1]])	
			translate_back = tf.constant([[1,0,0,-translate_val],
										[0,1,0,-translate_val],
										[0,0,1,-translate_val],
										[0,0,0,1]])	
			rotate = tf.constant([[math.cos(rad),-math.sin(rad),0,0],
								[math.sin(rad),math.cos(rad),0,0],
								[0,0,1,0],
								[0,0,0,1]])	
			scale = tf.constant([[scale,0,0,0],
								[0,scale,0,0],
								[0,0,scale,0],
								[0,0,0,1]])
								
			composed = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.matmul( \
						scale, translate_back), rotate), translate_forward)
			
			input_4d = tf.concat([input, tf.ones([input.shape[0],input.shape[1],input.shape[2],1], dtype=tf.dtypes.float32)], axis=-1)
			translated_4d = tf.reshape(tf.linalg.matmul(composed,tf.expand_dims(input_4d, axis=-1)), [input_4d.shape[0],input_4d.shape[1],input_4d.shape[2],input_4d.shape[3]])
			translated = tf.slice(translated_4d, [0,0,0,0], [translated_4d.shape[0],translated_4d.shape[1],translated_4d.shape[2],translated_4d.shape[3]-1])
			
			#tf.print("translate_forward:\n", composed)
			#tf.print("rotate:\n", composed)
			#tf.print("translate_back:\n", composed)
			#tf.print("composed:\n", composed)
			#tf.print("input:", input.shape, "\n", input)
			#tf.print("translated:", translated.shape, "\n", translated)
			
			return translated
		
		else:
			return input

class FlattenDims(tf.keras.layers.Layer):
	"""
	Adds a learnable bias to each input.
	"""
	def __init__(self):
		super(FlattenDims, self).__init__()

	def call(self, input):
		return tf.reshape(input, (-1,input.shape[1],input.shape[2]*input.shape[3]))

class LearnedComplexPosEncoding(tf.keras.layers.Layer):
	"""
	Adds a learnable bias to each input.
	"""
	def __init__(self, ignore_zeros=False):
		super(LearnedComplexPosEncoding, self).__init__()
		self.ignore_zeros = ignore_zeros

	def build(self, input_shape):
		self.encodings = self.add_weight("encodings", shape=[1,input_shape[-2],input_shape[-1]])

	def call(self, input):
		if self.ignore_zeros:
			max_vals = tf.reduce_max(input, axis=-1, keepdims=True)
			encodings = (self.encodings * max_vals) / (max_vals + 0.00001)
		else:
			encodings = self.encodings
		output = input + encodings
		return output
		
class FeatureSumLayer(tf.keras.layers.Layer):
	"""
	Selects one attention block to pass forward
	"""
	def __init__(self):
		super(FeatureSumLayer, self).__init__()

	def call(self, input):
		output = tf.math.reduce_sum(input, axis=-2)
		return output
	
class MyDenseLayer(tf.keras.layers.Layer):
	"""
	A dense layer with a skip connection
	"""
	def __init__(self, num_outputs, activation=None, skip_connection=False, dropout=0.):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs
		self.activation = activation
		self.skip_connection = skip_connection
		self.dropout = dropout

	def build(self, input_shape):
		self.p_weights = self.add_weight("p_weights", shape=[self.num_outputs,input_shape[-1]]) 
		self.p_biases = self.add_weight("p_biases", shape=[1,self.num_outputs]) 
    
	def call(self, input):
		if self.activation != None:
			output = self.activation(tf.matmul(input,tf.transpose(self.p_weights)) + self.p_biases)
		else:
			output =  tf.matmul(input,tf.transpose(self.p_weights)) + self.p_biases
		output = tf.nn.dropout(output, self.dropout)
		if self.skip_connection:
			return output + input
		else:
			return output
		
class MyMultiAttentionLayer(tf.keras.layers.Layer):
	"""
	Multi-head scaled dot-product attention layer
	"""	
	def __init__(self, num_outputs, param_number, multi_heads, activation=None, skip_connection=False, dropout=0.):
		super(MyMultiAttentionLayer, self).__init__()
		self.num_outputs = num_outputs
		self.param_number = param_number
		self.activation = activation
		self.multi_heads = multi_heads
		self.skip_connection = skip_connection
		self.dropout = dropout

	def build(self, input_shape):
		self.q_weights = self.add_weight("q_weights", shape=[self.multi_heads,self.param_number,input_shape[-1]]) 
		self.q_biases = self.add_weight("q_biases", shape=[self.multi_heads,1,self.param_number]) 

		self.k_weights = self.add_weight("k_weights", shape=[self.multi_heads,self.param_number,input_shape[-1]]) 
		self.k_biases = self.add_weight("k_biases", shape=[self.multi_heads,1,self.param_number]) 

		self.v_weights = self.add_weight("v_weights", shape=[self.multi_heads,self.param_number,input_shape[-1]]) 
		self.v_biases = self.add_weight("v_biases", shape=[self.multi_heads,1,self.param_number]) 

		self.l_weights = self.add_weight("l_weights", shape=[self.num_outputs,self.param_number*self.multi_heads])#shape=[self.num_outputs,input_shape[-2],self.param_number*self.multi_heads]) 
		self.l_biases = self.add_weight("l_biases", shape=[1,self.num_outputs]) 

	def call(self, input):

		exp_input = tf.tile(tf.expand_dims(input, -3), [1,self.multi_heads,1,1])

		query = tf.matmul(exp_input,tf.transpose(self.q_weights, perm=[0,2,1])) + self.q_biases
		key = tf.matmul(exp_input,tf.transpose(self.k_weights, perm=[0,2,1])) + self.k_biases
		value = tf.matmul(exp_input,tf.transpose(self.v_weights, perm=[0,2,1])) + self.v_biases

		scores = tf.nn.softmax(tf.matmul(query,tf.transpose(key, perm=[0,1,3,2]))/tf.math.sqrt(float(self.param_number)), axis=2)
		sum = tf.transpose(tf.matmul(scores, value), perm=[0,2,1,3])

		concat = tf.reshape(sum,[-1,sum.shape[-3],self.param_number*self.multi_heads])

		output = tf.matmul(concat,tf.transpose(self.l_weights)) + self.l_biases

		output = tf.nn.dropout(output, self.dropout)

		if self.skip_connection:
			skip_output = output + input
		else: 
			skip_output = output

		if self.activation != None:
			return self.activation(skip_output)
		else:
			return skip_output

class AttentionNetwork():
	"""
	Attention model for skeleton based activity recognition.
	"""	
	def __init__(self, batch_shape, block_size, mlp_hidden, num_outputs, \
			param_number, multi_heads, activation=tf.nn.relu, skip_connections=True):
		
		self.model = tf.keras.models.Sequential()
		self.model.add(ModifySkeletons())
		self.model.add(FlattenDims())
		self.model.add(MyDenseLayer(block_size))
		self.model.add(LearnedComplexPosEncoding(ignore_zeros=True))
		for layer in range(LAYERS):
			self.model.add(MyMultiAttentionLayer(block_size, param_number, multi_heads, \
					activation=activation, skip_connection=skip_connections, dropout=DROPOUT))
			self.model.add(tf.keras.layers.LayerNormalization(axis=-1))
			self.model.add(MyDenseLayer(block_size, activation=activation, skip_connection=skip_connections, dropout=DROPOUT))
			self.model.add(tf.keras.layers.LayerNormalization(axis=-1))
		self.model.add(FeatureSumLayer())
		self.model.add(MyDenseLayer(mlp_hidden))
		self.model.add(MyDenseLayer(CLASSES))
		self.model.add(tf.keras.layers.Activation('softmax'))
		
		self.model.compile(
		optimizer=OPTIMIZER,
		loss='sparse_categorical_crossentropy',
		metrics=['sparse_categorical_accuracy'])

		self.model.build(batch_shape)
		self.model.summary()
		
	def load_model(self, checkpoint_dir):
		print("loading model...")
		latest = tf.train.latest_checkpoint(checkpoint_dir)
		self.model.load_weights(latest)
		
	def train(self, epoch_num, epoch_steps, callback=None):
	
		self.model.fit(
		x_train.astype(np.float32), y_train.astype(np.float32),
		epochs=epoch_num,
		callbacks=[callback],
		steps_per_epoch=epoch_steps,
		validation_data=(x_test.astype(np.float32), y_test.astype(np.float32)),
		validation_freq=1
		)

# ---------- Constants ----------
DATA_FILE = '../nturgb_d_npy/'
SEED = 1
TRIAN_SIZE = 0.8
TEST_SIZE = 0.2
PAD_CLIP_LENGTH = 128

EPCOHS = 1000
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DROPOUT = 0.

MLP_UNITS = 256

BLOCK_SIZE = 128 	# > sample size
LAYERS = 8			
HEADS = 6		
PARAMETERS = 32  	# > block size / heads

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
INITIALIZER = tf.keras.initializers.GlorotUniform(SEED)
ACTIVATION_F =  tf.nn.relu

checkpoint_path = "checkpoints_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# ---------- Flags ----------

LOAD_MODEL = False
SCALE_CLIPS = True
SINGLE_POSE = False
FIRST_CLASSES = True


# ---------- Setup ----------

random.seed(SEED)
train_set, test_set = load_ntu(DATA_FILE, train_size=TRIAN_SIZE, test_size=TEST_SIZE, first_classes=FIRST_CLASSES, single_pose=SINGLE_POSE)

x_train, y_train = get_vals_labels(train_set)
x_test, y_test = get_vals_labels(test_set)

clip_lengths = [x.shape[0] for x in x_train]
print("Mean Clip Length:", round(np.mean(clip_lengths),2), "\tShortest Clip Length:", min(clip_lengths), "\tLongest Clip Length:", max(clip_lengths))

#set all clips to the same number of frames
normalize_clip_length(x_train, PAD_CLIP_LENGTH)
normalize_clip_length(x_test, PAD_CLIP_LENGTH)

#convert to numpy arrays
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train) - 1
y_test = np.array(y_test) - 1

#swap y and z coordinates so z is vertical
switch_yz_array = [[1,0,0],
					[0,0,1],
					[0,1,0]]
					

print(x_train.shape)
x_train = np.reshape(np.matmul(switch_yz_array, np.expand_dims(x_train, axis=-1)), x_train.shape)
print(x_train.shape)

#x_train = np.reshape(x_train, (-1,x_train.shape[1],x_train.shape[2]*x_train.shape[3]))
#x_test = np.reshape(x_test, (-1,x_test.shape[1],x_test.shape[2]*x_test.shape[3]))

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

EPOCH_STEPS = 	math.floor(x_train.shape[0]/BATCH_SIZE)
CLASSES =       int(np.max(y_train)-np.min(y_train))+1
BATCH_SHAPE =   tuple([BATCH_SIZE,x_train.shape[1],x_train.shape[2],x_train.shape[3]])#(int(x_train.shape[0]/EPOCH_STEPS),x_train.shape[1],x_train.shape[2])


"""# ---------- Aug Tests ----------

import matplotlib.pyplot as plt

#batch = np.array([[1,1,1],
#					[1,0,0],
#					[0,1,0],
#					[0,0,1]], dtype=float)

batch = x_train[0:5]

print("Batch Shape:", batch.shape)

matplotlib_clip_vid(batch[0])

print(batch[0].shape)

mod_skel = ModifySkeletons()
batch_mod = np.array(mod_skel(batch, training=True))

matplotlib_clip_vid(batch_mod[0])

print(batch_mod[0].shape)

batch_plt = np.transpose(batch_mod[0][0])	
	
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(batch_plt[0], batch_plt[1], batch_plt[2])		
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)		
plt.show()


_ = 1/0
"""

# ---------- Training ----------

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

network = AttentionNetwork(BATCH_SHAPE, BLOCK_SIZE, MLP_UNITS, BLOCK_SIZE, \
			PARAMETERS, HEADS, activation=ACTIVATION_F, skip_connections=True)
	
if LOAD_MODEL:
	network.load_model(checkpoint_dir)
else:
	network.model.save_weights(checkpoint_path.format(epoch=0))
			
network.train(EPCOHS, EPOCH_STEPS, callback=cp_callback)








		