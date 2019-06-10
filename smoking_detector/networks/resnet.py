from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50

def resnet() -> Model:

  num_classes = 2
  model = Sequential()

  model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
  model.add(Flatten())
  model.add(BatchNormalization())
  model.add(Dense(1024, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(2048, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(num_classes , activation='softmax'))

  model.layers[0].trainable = True
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model
