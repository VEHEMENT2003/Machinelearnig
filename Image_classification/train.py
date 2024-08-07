import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configure logging for the training script
logging.basicConfig(filename='train.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info('Starting model training process')
    try:
        # Initialize InceptionV3 model with pre-trained weights
        base_model = InceptionV3(weights='imagenet', include_top=False)

        # Add custom layers on top of InceptionV3
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)  # Updated to 10 classes

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze all layers in the base model
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Data augmentation and data generator
        train_datagen = ImageDataGenerator(rescale=0.2, horizontal_flip=True)
        valid_datagen = ImageDataGenerator(rescale=0.2)

        train_dir = os.path.join('Datasets', 'train')
        valid_dir = os.path.join('Datasets', 'valid')

        training_set = train_datagen.flow_from_directory(
            train_dir,
            target_size=(299, 299),
            batch_size=32,
            class_mode='categorical'
        )

        validation_set = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(299, 299),
            batch_size=32,
            class_mode='categorical'
        )

        # Callbacks
        checkpoint = ModelCheckpoint('model/model_inception.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

        # Train the model
        model.fit(
            training_set,
            steps_per_epoch=training_set.samples // training_set.batch_size,
            epochs=10,
            validation_data=validation_set,
            validation_steps=validation_set.samples // validation_set.batch_size,
            callbacks=[checkpoint, early_stopping]
        )

        logging.info('Model training completed successfully')

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    train_model()
