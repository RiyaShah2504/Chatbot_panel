"""
Application configuration with lazy loading and production optimization
"""
import os
from datetime import timedelta
from pathlib import Path


def configure_app(app):
    """Configure Flask application with proper settings"""

    # SECURITY CONFIGURATION
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        if os.getenv('FLASK_ENV') == 'production':
            raise ValueError("❌ SECRET_KEY must be set in production environment")
        else:
            SECRET_KEY = 'dev-secret-key-for-local-development-only'
            print("  Using development SECRET_KEY")

    app.config['SECRET_KEY'] = SECRET_KEY

    # ENVIRONMENT CONFIGURATION
    flask_env = os.getenv('FLASK_ENV', 'production')

    if flask_env == 'production':
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
        print(" Running in PRODUCTION mode")
    else:
        app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        app.config['TESTING'] = False
        print("  Running in DEVELOPMENT mode")

    # SESSION CONFIGURATION
    session_lifetime = int(os.getenv('PERMANENT_SESSION_LIFETIME', '2592000'))  # 30 days
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=session_lifetime)
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    # DATABASE CONFIGURATION
    DATABASE_URL = os.getenv('DATABASE_URL')

    if not DATABASE_URL:
        # Fallback for development
        DATABASE_URL = 'mysql+pymysql://root:root@localhost:3306/chatbot_panel'
        print("️  WARNING: Using local development database")
    else:
        print("✓ Using configured database")

    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Connection pool settings
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': int(os.getenv('SQLALCHEMY_POOL_SIZE', 10)),
        'pool_recycle': int(os.getenv('SQLALCHEMY_POOL_RECYCLE', 3600)),
        'pool_pre_ping': os.getenv('SQLALCHEMY_POOL_PRE_PING', 'True').lower() == 'true',
        'max_overflow': int(os.getenv('SQLALCHEMY_MAX_OVERFLOW', 20)),
        'pool_timeout': 30,
        'connect_args': {'connect_timeout': 10}
    }

    # FILE STORAGE CONFIGURATION
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DATA_FOLDER = os.getenv('BASE_DATA_FOLDER', os.path.join(BASE_DIR, 'data'))

    # Ensure directories exist
    os.makedirs(BASE_DATA_FOLDER, exist_ok=True)

    app.config['BASE_DATA_FOLDER'] = BASE_DATA_FOLDER
    app.config['USER_DATA_FOLDER'] = os.getenv(
        'USER_DATA_FOLDER',
        os.path.join(BASE_DATA_FOLDER, 'users')
    )
    app.config['UPLOAD_FOLDER'] = os.getenv(
        'UPLOAD_FOLDER',
        os.path.join(BASE_DATA_FOLDER, 'uploads')
    )

    # Create upload directories
    os.makedirs(app.config['USER_DATA_FOLDER'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # File size limits
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB

    print(f"✓ Data folder: {BASE_DATA_FOLDER}")

    # EMAIL CONFIGURATION
    app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 465))
    app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'False').lower() == 'true'
    app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'True').lower() == 'true'
    app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_FROM_ADDRESS')

    # Verify email configuration
    if app.config['MAIL_USERNAME']:
        print(f"✓ Email configured: {app.config['MAIL_USERNAME']}")
    else:
        print("⚠️  Email not configured")

    # PERFORMANCE & CACHING
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year for static files

    return app