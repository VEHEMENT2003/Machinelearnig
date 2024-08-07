import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configure logging for the evaluation script
logging.basicConfig(filename='evaluate.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model():
    logging.info('Starting model evaluation process')
    try:
        model = load_model('model/model_inception.h5')

        valid_datagen = ImageDataGenerator(rescale=0.2)
        valid_dir = os.path.join('Datasets', 'valid')

        validation_set = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(299, 299),
            batch_size=32,
            class_mode='categorical'
        )

        # Evaluate the model
        loss, accuracy = model.evaluate(
            validation_set,
            steps=validation_set.samples // validation_set.batch_size
        )

        logging.info(f'Model evaluation completed successfully with accuracy: {accuracy} and loss: {loss}')

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    evaluate_model()
