import logging
import json
import os

def get_logger(name):
    """Configures and returns a logger that writes to both console and logs.txt"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if instantiated multiple times
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler('logs.txt')
        file_handler.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def load_config(config_path):
    """Loads a JSON configuration file."""
    with open(config_path, 'r') as file:
        return json.load(file)