
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

# Dataset path
data_path = pathlib.Path('Images')

# Load training dataset
train_ds = image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

# Load validation dataset
test_ds = image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=16
)

# Load pretrained DenseNet121
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Build CNN
cnn = Sequential()

cnn.add(Rescaling(1./255))
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dropout(0.75))
cnn.add(Dense(120, activation='softmax'))

# Compile model
cnn.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.00001),
    metrics=['accuracy']
)

# Train model
hist = cnn.fit(
    train_ds,
    epochs=200,
    validation_data=test_ds,
    verbose=2
)

# Evaluate model
print('Accuracy:', cnn.evaluate(test_ds, verbose=2)[1] * 100)

# Save trained model
cnn.save('cnn_for_stanford_dogs.h5')

# Save class names
import pickle

f = open('dog_species_names.txt', 'wb')
pickle.dump(train_ds.class_names, f)
f.close()

# Plot accuracy graph
import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

# Plot loss graph
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()
