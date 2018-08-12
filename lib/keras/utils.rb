module Utils
  extend self

  def to_categorical(y, **args)
    pyfrom 'keras.utils', import: 'to_categorical'
    to_categorical(y, **args)
  end
end
