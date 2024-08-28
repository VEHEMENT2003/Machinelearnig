import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Configure logging for the training script
logging.basicConfig(filename='logs/train.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info('Starting model training process')
    try:
        # Data preparation
        train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
        valid_datagen = ImageDataGenerator(rescale=1.0/255.0)

        train_dir = os.path.join('Datasets', 'train')
        valid_dir = os.path.join('Datasets', 'valid')

        training_set = train_datagen.flow_from_directory(
            train_dir,
            target_size=(299, 299),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )

        validation_set = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(299, 299),
            batch_size=32,
            class_mode='categorical'
        )

        # Load InceptionV3 model with pre-trained weights
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

        # Add custom layers on top of InceptionV3
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(len(training_set.class_indices), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the base model
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the model
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        logging.info('Model compiled successfully')

        # Train the model
        history = model.fit(
            training_set,
            validation_data=validation_set,
            epochs=5,
            steps_per_epoch=training_set.samples // training_set.batch_size,
            validation_steps=validation_set.samples // validation_set.batch_size
        )
        
        # Save the model
        model_path = 'model/model_inception.keras'
        model.save(model_path)
        logging.info(f'Model saved to {model_path}')

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    train_model()
