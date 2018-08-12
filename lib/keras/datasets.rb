module Datasets
  extend self

  def cifar10
    pyfrom 'keras.datasets', import: 'cifar10'
    cifar10.load_data
  end
end
