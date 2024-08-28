import os
import logging
import train
import evaluate

# Configure logging
logging.basicConfig(filename='logs/main.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Ensure necessary directories exist
    if not os.path.exists('model'):
        os.makedirs('model')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    logging.info('Starting the training process')
    # Train the model
    train.train_model()

    logging.info('Starting the evaluation process')
    # Evaluate the model
    evaluate.evaluate_model()

except Exception as e:
    logging.error(f"An error occurred: {e}")
    raise
