import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure logging for the evaluation script
logging.basicConfig(filename='logs/evaluate.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model():
    logging.info('Starting model evaluation process')
    try:
        # Load the trained model
        model_path = 'model/model_inception.keras'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = load_model(model_path)
        logging.info('Model loaded successfully')

        # Data preparation for evaluation
        valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
        valid_dir = os.path.join('Datasets', 'valid')

        validation_set = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(299, 299),
            batch_size=32,
            class_mode='categorical'
        )

        # Evaluate the model
        loss, accuracy = model.evaluate(validation_set, steps=validation_set.samples // validation_set.batch_size)
        logging.info(f'Model evaluation results - Loss: {loss}, Accuracy: {accuracy}')

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    evaluate_model()
