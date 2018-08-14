require './lib/keras'

python_version('~/miniconda3/bin/python')

array = Keras::Datasets.cifar10
x_train = array[0][0].astype('float32') / 255
y_train = Keras::Utils::to_categorical array[0][1], num_classes: 10
x_test = array[1][0].astype('float32') / 255
y_test = Keras::Utils::to_categorical array[1][1], num_classes: 10

model = Keras::Model.sequential
model.add(Keras::Convolutional.conv2D(32, [3, 3], padding: 'same', input_shape: x_train[0].shape))
model.add(Keras::Core.activation('relu'))
model.add(Keras::Convolutional.conv2D(32, [3, 3]))
model.add(Keras::Core.activation('relu'))
model.add(Keras::Pooling.max_pooling2D(pool_size: [2, 2]))
model.add(Keras::Core.dropout(0.25))

model.add(Keras::Convolutional.conv2D(64, [3, 3], padding: 'same'))
model.add(Keras::Core.activation('relu'))
model.add(Keras::Convolutional.conv2D(64, [3, 3], padding: 'same'))
model.add(Keras::Core.activation('relu'))
model.add(Keras::Pooling.max_pooling2D(pool_size: [2, 2]))
model.add(Keras::Core.dropout(0.25))

model.add(Keras::Core.flatten())
model.add(Keras::Core.dense(512))
model.add(Keras::Core.activation('relu'))
model.add(Keras::Core.dropout(0.5))
model.add(Keras::Core.dense(10))
model.add(Keras::Core.activation('softmax'))

opt = Keras::Optimizers.rmsprop(lr: 0.0001, decay: (10**-6).to_f)
model.compile(loss: 'categorical_crossentropy', optimizer: opt, metrics: ['accuracy'])

model.fit(x_train, y_train, batch_size: 32, epochs: 1, validation_data: [x_test, y_test], shuffle: true)

model.save('keras_cifar10_trained_model.h5')

scores = model.evaluate(x_test, y_test, verbose: 1)
puts 'Test loss:' + scores[0].to_s
puts 'Test accuracy:' + scores[1].to_s
