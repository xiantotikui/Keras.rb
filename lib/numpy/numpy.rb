module Numpy
  extend self
  pyimport :numpy

  def array(x)
    numpy.array(x)
  end

  def asarray(x)
    numpy.asarray(x)
  end

  def log(x)
    numpy.log(x)
  end

  def exp(x)
    numpy.exp(x)
  end

  def sum(x)
    numpy.sum(x)
  end

  def multinomial(v0, x, v1)
    numpy.random.multinomial(v0, x, v1)
  end

  def argmax(x)
    numpy.argmax(x)
  end

  def amax(x)
    numpy.amax(x)
  end

  def reshape(a, newshape)
    numpy.reshape(a, newshape)
  end

  def zeros(x)
    numpy.zeros(x)
  end

  def concatenate(x)
    numpy.concatenate(x)
  end

  def append(x0, *x1)
    numpy.append(x0, *x1)
  end
end
