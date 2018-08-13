module Keras::Convolutional
  extend self
  pyfrom 'keras.layers', import: 'Conv1D'
  pyfrom 'keras.layers', import: 'Conv2D'
  pyfrom 'keras.layers', import: 'Conv3D'
  pyfrom 'keras.layers', import: 'SeparableConv1D'
  pyfrom 'keras.layers', import: 'SeparableConv2D'
  pyfrom 'keras.layers', import: 'Conv2DTranspose'
  pyfrom 'keras.layers', import: 'Cropping1D'
  pyfrom 'keras.layers', import: 'Cropping2D'
  pyfrom 'keras.layers', import: 'Cropping3D'
  pyfrom 'keras.layers', import: 'UpSampling1D'
  pyfrom 'keras.layers', import: 'UpSampling2D'
  pyfrom 'keras.layers', import: 'UpSampling3D'
  pyfrom 'keras.layers', import: 'ZeroPadding1D'
  pyfrom 'keras.layers', import: 'ZeroPadding2D'
  pyfrom 'keras.layers', import: 'ZeroPadding3D'

  def conv1D(filters, kernel_size, **args)
    Conv1D.new(filters, kernel_size, **args)
  end

  def conv2D(filters, kernel_size, **args)
    Conv2D.new(filters, kernel_size, **args)
  end

  def conv3D(filters, kernel_size, **args)
    Conv3D.new(filters, kernel_size, **args)
  end

  def separable_conv1D(filters, kernel_size, **args)
    SeparableConv1D.new(filters, kernel_size, **args)
  end

  def separable_conv2D(filters, kernel_size, **args)
    SeparableConv2D.new(filters, kernel_size, **args)
  end

  def conv2D_transpose(filters, kernel_size, **args)
    Conv2DTranspose.new(filters, kernel_size, **args)
  end

  def cropping1D(cropping: [1, 1], **args)
    Cropping1D.new(cropping: cropping, **args)
  end

  def cropping2D(cropping: [[0, 0], [0, 0]], **args)
    Cropping2D.new(cropping: cropping, **args)
  end

  def cropping3D(cropping: [[1, 1], [1, 1], [1, 1]], **args)
    Cropping3D.new(cropping: cropping, **args)
  end

  def upsampling1D(size: 2, **args)
    UpSampling1D.new(size: size, **args)
  end

  def upsampling2D(size: [2 ,2], **args)
    UpSampling2D.new(size: size, **args)
  end

  def upsampling3D(size: [2 ,2, 2], **args)
    UpSampling3D.new(size: size, **args)
  end

  def zero_padding1D(padding: 1, **args)
    ZeroPadding1D.new(padding: padding, **args)
  end

  def zero_padding2D(padding: [1, 1], **args)
    ZeroPadding2D.new(padding: padding, **args)
  end

  def zero_padding3D(padding: [1, 1, 1], **args)
    ZeroPadding3D.new(padding: padding, **args)
  end
end
