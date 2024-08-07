
import os
import train
import evaluate
import logging

# Configure logging for the main script
logging.basicConfig(filename='main.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Ensure necessary directories exist
    if not os.path.exists('model'):
        os.makedirs('model')
        logging.info('Created directory: model')

    logging.info('Starting the training and evaluation process')

    try:
        # Train the model
        train.train_model()
        logging.info('Model training completed successfully')

        # Evaluate the model
        evaluate.evaluate_model()
        logging.info('Model evaluation completed successfully')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
