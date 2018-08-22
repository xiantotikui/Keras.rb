require 'pycall/import'
include PyCall::Import

module Keras; end

require './lib/numpy/numpy'

def keras_import(modules)
  require './lib/keras/models' if modules.include? 'Models'
  require './lib/keras/core' if modules.include? 'Core'
  require './lib/keras/optimizers' if modules.include? 'Optimizers'
  require './lib/keras/datasets' if modules.include? 'Datasets'
  require './lib/keras/convolutional' if modules.include? 'Convolutional'
  require './lib/keras/pooling' if modules.include? 'Pooling'
  require './lib/keras/utils' if modules.include? 'Utils'
  require './lib/keras/locally_connected' if modules.include? 'LocallyConnected'
  require './lib/keras/recurrent' if modules.include? 'Recurrent'
  require './lib/keras/embedding' if modules.include? 'Embedding'
  require './lib/keras/merge' if modules.include? 'Merge'
  require './lib/keras/advanced_activations' if modules.include? 'AdvancedActivations'
  require './lib/keras/normalization' if modules.include? 'Normalization'
  require './lib/keras/wrappers' if modules.include? 'Wrappers'
  require './lib/keras/callbacks' if modules.include? 'Callbacks'
end
