require './lib/keras'

test = Model.sequential
p test.add(Core.dense(100, 1000))
p test.add(Core.activation('relu'))
p test.add(Core.dropout(0.1))
p test.add(Core.reshape([10, 10]))

require 'pycall/import'
include PyCall::Import
pyimport :keras
sgd = Optimizers.sgd
p Datasets.cifar10

p test.compile(sgd, 'mean_squared_error')
