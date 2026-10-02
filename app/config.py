"""
Central configuration — reads from environment variables with defaults.
Create a .env file in the project root to override any value.
"""

import os

API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")
APP_VERSION = "1.0.0"
APP_TITLE   = "Sanjeevani — Fatty Liver Risk Assessment API"
