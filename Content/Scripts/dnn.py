import tensorflow as tf
import numpy as np
import logging
import unreal_engine as ue
import upypip as pip
from pathlib import Path
#from PIL import Image

class DNN:
  """
  Convolutional Neural Network model.
  """

  def __init__(self, folder, num_actions, observation_shape, params={}, verbose=False):
    """
    Initialize the CNN model with a set of parameters.
    Args:
      params: a dictionary containing values of the models' parameters.
    """
    self.scripts_path = ue.get_content_dir() + "Scripts"
    self.model_directory = self.scripts_path + "/models" + "/" + folder

    #Create a folder if we dont have one already
    Path(self.model_directory).mkdir(parents=True, exist_ok=True)

    self.modemodel_loaded = False

    self.model_path = self.model_directory + "/model.ckpt"
    self.verbose = verbose
    self.num_actions = num_actions

    # observation shape will be a tuple
    self.observation_shape = observation_shape[0]
    logging.info('Initialized with params: {}'.format(params))

    #hyperparameters
    self.use_cnn = params['use_cnn']
    self.lr = params['lr']
    self.reg = params['reg']
    self.hidden_layers = params['hidden_layers']
    self.conv_layers = params['conv_layers']
    self.image_width = params['image_width']
    self.image_height = params['image_height']
    self.color_channels = params['color_channels']
    self.W = []
    self.b = []
    self.conv = []
    self.fc = []
    self.use_maxpooling = params['use_maxpooling']
    self.session = self.create_model()


  def add_placeholders(self):
    input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape))
    labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
    actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))

    return input_placeholder, labels_placeholder, actions_placeholder

  def cnn(self, input_obs):
    ue.log(str('CNN created.'))
    input_obs = tf.reshape(input_obs, shape=[-1, self.image_height, self.image_width, self.color_channels], name="reshapedInput")

    if self.conv_layers[0][4] == 1:
        conv1 = tf.layers.conv2d(inputs = input_obs, filters = int(self.conv_layers[0][0]), kernel_size = int(self.conv_layers[0][1]), strides = int(self.conv_layers[0][2]), padding = self.conv_layers[0][3], activation = tf.nn.relu, name="conv1")
    else:
        conv1 = tf.layers.conv2d(inputs = input_obs, filters = int(self.conv_layers[0][0]), kernel_size = int(self.conv_layers[0][1]), strides = int(self.conv_layers[0][2]), padding = self.conv_layers[0][3], activation = None, name="conv1")
      
    if self.use_maxpooling == 1:
        conv1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 1, name="maxpool1")
      
    self.conv.append(conv1)

    for i in range(len(self.conv_layers)-1):
        convName = "conv" + str(i+2)
        maxPoolName = "maxpool" + str(i+2)
        current_filters = int(self.conv_layers[i+1][0])
        current_kernels = int(self.conv_layers[i+1][1])
        current_strides = int(self.conv_layers[i+1][2])
        current_padding = self.conv_layers[i+1][3]

        if self.conv_layers[i+1][4] == 1:
            conv = tf.layers.conv2d(inputs = self.conv[i], filters = current_filters, kernel_size = current_kernels, strides = current_strides, padding = current_padding, activation = tf.nn.relu, name=convName)
        else:
            conv = tf.layers.conv2d(inputs = self.conv[i], filters = current_filters, kernel_size = current_kernels, strides = current_strides, padding = current_padding, activation = None, name=convName)

        if self.use_maxpooling == 1:
            conv = tf.layers.max_pooling2d(inputs = conv, pool_size = [2,2], strides = 1, name=maxPoolName)

        self.conv.append(conv)

    finalConv = tf.contrib.layers.flatten(self.conv[len(self.conv)-1])
    fc1 = tf.layers.dense(finalConv, self.hidden_layers[0], name="fc1", activation = tf.nn.relu)
    self.fc.append(fc1)

    for i in range(len(self.hidden_layers)-1):
        fcName = "fc" + str(i+2)
        fc = tf.layers.dense(self.fc[i], self.hidden_layers[i+1], name=fcName, activation = tf.nn.relu)
        self.fc.append(fc)

    out = tf.layers.dense(self.fc[len(self.fc)-1], self.num_actions, name="out")
    
    return out

  def dnn(self, input_obs):
    with tf.name_scope("Layer1") as scope:
      W1shape = [self.observation_shape, self.hidden_layers[0]]
      self.W.append(tf.get_variable("W1", shape=W1shape,))
      b1shape = [1, self.hidden_layers[0]]
      self.b.append(tf.get_variable("b1", shape=b1shape, initializer = tf.constant_initializer(0.0)))

    for i in range(len(self.hidden_layers)-1):
        scopeName = "Layer" + str(i+2)
        WName = "W" + str(i+2)
        bName = "b" + str(i+2)
        with tf.name_scope(scopeName) as scope:
            Wshape = [self.hidden_layers[i], self.hidden_layers[i+1]]
            self.W.append(tf.get_variable(WName, shape=Wshape,))
            bshape = [1, self.hidden_layers[i+1]]
            self.b.append(tf.get_variable(bName, shape=bshape, initializer = tf.constant_initializer(0.0)))
    
    scopeName = "Layer" + str(len(self.hidden_layers)+1)
    WName = "W" + str(len(self.hidden_layers)+1)
    bName = "b" + str(len(self.hidden_layers)+1)
    with tf.name_scope(scopeName) as scope:
      Wshape = [self.hidden_layers[len(self.hidden_layers)-1], self.hidden_layers[len(self.hidden_layers)-1]]
      self.W.append(tf.get_variable(WName, shape=Wshape,))
      b4shape = [1, self.hidden_layers[len(self.hidden_layers)-1]]
      self.b.append(tf.get_variable(bName, shape=b4shape, initializer = tf.constant_initializer(0.0)))

    with tf.name_scope("OutputLayer") as scope:
      Ushape = [self.hidden_layers[len(self.hidden_layers)-1], self.num_actions]
      self.U = tf.get_variable("U", shape=Ushape)
      boshape = [1, self.num_actions]
      self.bo = tf.get_variable("bo", shape=boshape, initializer = tf.constant_initializer(0.0))

    xW = tf.matmul(input_obs, self.W[0])
    h = tf.tanh(tf.add(xW, self.b[0]))
    regCalc = tf.reduce_sum(tf.square(self.W[0]))
    for i in range(len(self.W)-1):
        xW = tf.matmul(h, self.W[i+1])
        h = tf.tanh(tf.add(xW, self.b[i+1]))
        regCalc += tf.reduce_sum(tf.square(self.W[i+1]))

    hU = tf.matmul(h, self.U)
    out = tf.add(hU, self.bo)
    regCalc += tf.reduce_sum(tf.square(self.U))

    reg = self.reg * regCalc
    
    ue.log('model values created')
    return out, reg

  def create_model(self):
    """
    The model definition.
    """
    tf.reset_default_graph()
    session = tf.Session()

    self.input_placeholder, self.labels_placeholder, self.actions_placeholder = self.add_placeholders()
    
    if self.use_cnn == 1:
        outputs = self.cnn(self.input_placeholder)
        self.predictions = outputs
        self.q_vals = tf.reduce_sum(tf.multiply(self.predictions, self.actions_placeholder), 1)
        self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)
    else:
        outputs, reg = self.dnn(self.input_placeholder)
        self.predictions = outputs
        self.q_vals = tf.reduce_sum(tf.multiply(self.predictions, self.actions_placeholder), 1)
        self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals)) + reg
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)

    self.train_op = optimizer.minimize(self.loss)

    self.saverino = tf.train.Saver()
    try:
        saver = tf.train.Saver()
        saver.restore(session, self.model_path)
        ue.log("model restored")
        #ue.log(str(session.run(self.W2)))#test values
    except:
        init = tf.initialize_all_variables()
        self.saverino = tf.train.Saver()
        session.run(init)
        ue.log('Created new model')

    ue.log('session created')
    return session

  def train_step(self, Xs, ys, actions):
    """
    Updates the CNN model with a mini batch of training examples.
    """
    loss, _, prediction_probs, q_values = self.session.run(
      [self.loss, self.train_op, self.predictions, self.q_vals],
      feed_dict = {self.input_placeholder: Xs,
                  self.labels_placeholder: ys,
                  self.actions_placeholder: actions
                  })

  def predict(self, observation):
    """
    Predicts the rewards for an input observation state. 
    Args:
      observation: a numpy array of a single observation state
    """

    loss, prediction_probs = self.session.run(
      [self.loss, self.predictions],
      feed_dict = {self.input_placeholder: observation,
                  self.labels_placeholder: np.zeros(len(observation)),
                  self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                  })

    return prediction_probs

  def saveModel(self, inputQ, actionQ):
    path = self.saverino.save(self.session, self.model_path)
    #ue.log(str(self.session.run(self.W2)))#test values
    ue.log("Saved model: "+str(self.model_path))
    pass

