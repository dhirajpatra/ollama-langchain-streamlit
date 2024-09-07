# config.py
import os

# Base directory of the application
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # Database URL from environment variable or default to SQLite in memory for development
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password123@localhost/lcnc_db_dev")
    
    # Disable modification tracking to save resources
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Add any other configuration options as needed

# Define a settings object for use in main.py
settings = Config()
