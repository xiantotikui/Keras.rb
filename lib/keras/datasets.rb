module Datasets
  pyimport :keras
  extend self
  
  attr_reader :x_train, :y_train, :x_test, :y_test

  def cifar10
    ret = -> { keras.datasets.cifar10.load_data }
    arr = ret.()
    @x_train = arr[0]
    @y_train = arr[1]
    @x_test = arr[2]
    @y_test = arr[3]
  end
end
