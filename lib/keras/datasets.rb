module Keras::Datasets
  extend self

  def cifar10
    pyfrom 'keras.datasets', import: 'cifar10'
    cifar10.load_data
  end

  def cifar100(label_mode: 'fine', **args)
    pyfrom 'keras.datasets', import: 'cifar100'
    cifar100.load_data(label_mode: label_mod, **args)
  end

  def imdb(path: "imdb.npz", **args)
    pyfrom 'keras.datasets', import: 'imdb'
    imdb.load_data(path: path, **args)
  end

  def reuters(path: "reuters.npz", **args)
    pyfrom 'keras.datasets', import: 'reuters'
    reuters.load_data(path: path, **args)
  end

  def mnist
    pyfrom 'keras.datasets', import: 'mnist'
    mnist.load_data
  end

  def fashion_mnist
    pyfrom 'keras.datasets', import: 'fashion_mnist'
    fashion_mnist.load_data
  end

  def boston_housing
    pyfrom 'keras.datasets', import: 'boston_housing'
    boston_housing.load_data
  end
end
