require './lib/keras'

array = Datasets.cifar10
x_train = array[0][0].astype('float32') / 255
y_train = Utils::to_categorical array[0][1], num_classes: 10
x_test = array[1][0].astype('float32') / 255
y_test = Utils::to_categorical array[1][1], num_classes: 10

model = Model.sequential
model.add(Convolutional.conv2D(32, [3, 3], padding: 'same', input_shape: x_train[0].shape))
model.add(Core.activation('relu'))
model.add(Convolutional.conv2D(32, [3, 3]))
model.add(Core.activation('relu'))
model.add(Pooling.max_pooling2D([2, 2]))
model.add(Core.dropout(0.25))

model.add(Convolutional.conv2D(64, [3, 3], padding: 'same'))
model.add(Core.activation('relu'))
model.add(Convolutional.conv2D(64, [3, 3], padding: 'same'))
model.add(Core.activation('relu'))
model.add(Pooling.max_pooling2D([2, 2]))
model.add(Core.dropout(0.25))

model.add(Core.flatten())
model.add(Core.dense(512))
model.add(Core.activation('relu'))
model.add(Core.dropout(0.5))
model.add(Core.dense(10))
model.add(Core.activation('softmax'))

opt = Optimizers.rmsprop(learning_rate: 0.0001, decay: (10**-6).to_f)
model.compile(loss: 'categorical_crossentropy', optimizer: opt, metrics: ['accuracy'])

model.fit(x_train, y_train, batch_size: 32, epochs: 10, validation_data: [x_test, y_test], shuffle: true)

model.save('keras_cifar10_trained_model.h5')

scores = model.evaluate(x_test, y_test, verbose: 1)
puts 'Test loss:' + scores[0].to_s
puts 'Test accuracy:' + scores[1].to_s
