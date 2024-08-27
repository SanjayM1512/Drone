import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

# Define the number of classes based on the dataset
NUM_CLASSES = 4  # Update to match the number of classes

# Paths to your training dataset
train_data_dir = r'C:\Users\HP\OneDrive\Desktop\IN-CNN\Train'

# Data Augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loading training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # Multi-class classification
)

# Load InceptionV3 with pretrained weights
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add Dropout layer to prevent overfitting
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # Update to NUM_CLASSES

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers to prevent training on them
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up model checkpoints
checkpoint = ModelCheckpoint('best_model.keras', monitor='accuracy', save_best_only=True, mode='max')

# Convert data generator to tf.data.Dataset for better performance
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, IMG_SIZE[0], IMG_SIZE[1], 3], [None, NUM_CLASSES])
).prefetch(tf.data.AUTOTUNE)

# Train the model and print training accuracy
history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,
    callbacks=[checkpoint]
)

# Access training accuracy from the history object
train_accuracy = history.history['accuracy'][-1]  # Get the last training accuracy value
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

average_accuracy = sum(history.history['accuracy']) / len(history.history['accuracy'])
print(f'Average Training Accuracy: {average_accuracy * 100:.2f}%')
