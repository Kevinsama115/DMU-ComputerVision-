
import tensorflow.keras.datasets as ds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()

# Normalize images
x_train = x_train.astype('float32') / 255

# Select the first 15 images
x_train = x_train[0:15]
y_train = y_train[0:15]

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Display original images
plt.figure(figsize=(20, 2))

for i in range(15):
    plt.subplot(1, 15, i + 1)
    plt.imshow(x_train[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(class_names[int(y_train[i][0])])

plt.show()

# Image augmentation
batch_size = 4

generator = ImageDataGenerator(
    rotation_range=20.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

gen = generator.flow(
    x_train,
    y_train,
    batch_size=batch_size
)

# Generate augmented images
for a in range(3):
    img, label = next(gen)

    plt.figure(figsize=(8, 2))
    plt.suptitle('Generator trial ' + str(a + 1))

    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(img[i])
        plt.xticks([])
        plt.yticks([])
        plt.title(class_names[int(label[i][0])])

    plt.show()
