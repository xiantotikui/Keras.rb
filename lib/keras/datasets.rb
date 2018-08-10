module Datasets
  pyimport :keras
  extend self

  def cifar10
    ret = -> { keras.datasets.cifar10.load_data }
    ret.()
  end
end
