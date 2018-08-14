require 'pycall/import'
include PyCall::Import

def python_version(path)
  PyCall.init(path)
end

module Keras; end

require './lib/keras/model'
require './lib/keras/core'
require './lib/keras/optimizers'
require './lib/keras/datasets'
require './lib/keras/convolutional'
require './lib/keras/pooling'
require './lib/keras/utils'
require './lib/keras/locally_connected'
require './lib/keras/recurrent'
require './lib/keras/embedding'
require './lib/keras/merge'
require './lib/keras/advanced_activations'
require './lib/keras/normalization'
require './lib/keras/wrappers'
