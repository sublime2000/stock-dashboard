"""
WSGI configuration for PythonAnywhere deployment.
This file is used by PythonAnywhere to serve the Flask application.
"""

import sys
import os

# Add your project directory to the sys.path
# Replace 'yourusername' with your actual PythonAnywhere username
project_home = '/home/yourusername/stock-dashboard'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['FLASK_ENV'] = 'production'

# Import the Flask app
from app import app as application
