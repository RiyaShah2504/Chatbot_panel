import base64
import difflib
import io
import json
import os
import random
import re
import secrets
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path

import dns.resolver
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail, Message
from flask_talisman import Talisman
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Import database models
from database import ChatSession, create_session, log_message, get_session_messages
from database import SubscriptionPlan, Subscription, create_trial_subscription, initialize_subscription_plans
from database import db, User, Chatbot, QAPair, save_qa_pairs_to_db

# ✅ LAZY IMPORT: Only import utils when needed (not on startup)
# This prevents loading heavy ML models during app initialization

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


# APPLICATION FACTORY for better WSGI compatibility
def create_app():
    app = Flask(__name__)
    # Import configuration
    from config import configure_app
    configure_app(app)
    # Initialize database
    db.init_app(app)
    # Initialize other extensions
    mail = Mail(app)
    return app
app = create_app()

SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    if os.getenv('FLASK_ENV') == 'production':
        raise ValueError("SECRET_KEY must be set in production environment")
    else:
        SECRET_KEY = 'dev-secret-key-for-local-development-only'
app.config['SECRET_KEY'] = SECRET_KEY
print(secrets.token_urlsafe(32))

# Production-ready configuration
if os.getenv('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
else:
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Session Configuration
session_lifetime = int(os.getenv('PERMANENT_SESSION_LIFETIME', '2592000'))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=session_lifetime)
# Get DATABASE_URL from environment, with local fallback
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    # Fallback to local development database
    DATABASE_URL = 'mysql+pymysql://root:root@localhost:3306/chatbot_panel'
    print("  WARNING: Using local development database")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
if not app.config['SQLALCHEMY_DATABASE_URI']:
    raise ValueError("DATABASE_URL must be set in .env file")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:root@localhost:3306/chatbot_panel')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
    'max_overflow': 20,
    'pool_timeout': 30,
    'connect_args': {'connect_timeout': 10}
}

# Data folder configuration from environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_FOLDER = os.getenv('BASE_DATA_FOLDER', os.path.join(BASE_DIR, 'data'))
app.config['USER_DATA_FOLDER'] = os.getenv('USER_DATA_FOLDER', os.path.join(BASE_DATA_FOLDER, 'users'))
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', os.path.join(BASE_DATA_FOLDER, 'uploads'))

# Get max content length from env (in bytes)
max_content_mb = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))
app.config['MAX_CONTENT_LENGTH'] = max_content_mb

# File upload configuration from environment
allowed_ext_str = os.getenv('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif,svg')
ALLOWED_EXTENSIONS = set(ext.strip() for ext in allowed_ext_str.split(','))
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '2097152'))

# Email Configuration from environment
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', '465'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'False').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_FROM_ADDRESS')

# Logging for production
if app.config.get('DEBUG'):
    print(f"Mail Server: {app.config['MAIL_SERVER']}")
    print(f"Mail Port: {app.config['MAIL_PORT']}")
    print(f"Mail Username: {app.config['MAIL_USERNAME']}")
    print(f"Mail Configured: {bool(app.config['MAIL_USERNAME'])}")


def create_directory_structure():
    """Create all necessary directory structure with proper permissions"""
    directories = [
        app.config['USER_DATA_FOLDER'],
        app.config['UPLOAD_FOLDER'],
        os.path.join(BASE_DATA_FOLDER, 'temp'),
        os.path.join(BASE_DATA_FOLDER, 'backups')
    ]

    # Add log directory if configured
    log_file = os.getenv('LOG_FILE')
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            directories.append(log_dir)

    for directory in directories:
        try:
            os.makedirs(directory, mode=0o755, exist_ok=True)
            if app.config.get('DEBUG'):
                print(f"✓ Directory ready: {directory}")
        except Exception as e:
            print(f"✗ Failed to create {directory}: {e}")
            raise


create_directory_structure()

# Configure logging for production
if os.getenv('LOG_FILE'):
    import logging
    from logging.handlers import RotatingFileHandler

    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_file = os.getenv('LOG_FILE')
    max_bytes = int(os.getenv('LOG_MAX_BYTES', '10485760'))
    backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))

    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))
    handler.setLevel(getattr(logging, log_level))
    app.logger.addHandler(handler)
    app.logger.setLevel(getattr(logging, log_level))

db.init_app(app)
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

executor = ThreadPoolExecutor(max_workers=10)


# # SECURITY: Headers & HTTPS
# if os.getenv('FLASK_ENV') != 'development':
#     Talisman(app,
#         force_https=True,
#         strict_transport_security=True,
#         content_security_policy={
#             'default-src': "'self'",
#             'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https://cdnjs.cloudflare.com"],
#             'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
#             'img-src': ["'self'", "data:", "https:"],
#         }
#     )
#
#
# # SECURITY: CORS Configuration
# CORS(app, resources={
#     r"/api/*": {
#         "origins": os.getenv('CORS_ORIGINS', '*').split(','),
#         "methods": ["POST", "OPTIONS"]
#     },
#     r"/embed/*": {"origins": "*"}
# })
#
#
# # SECURITY: Rate Limiting
# limiter = Limiter(
#     app=app,
#     key_func=get_remote_address,
#     default_limits=["200 per day", "50 per hour"],
#     storage_uri="memory://"
# )
#
#
# # SECURITY: HTTPS Enforcement
# @app.before_request
# def enforce_https():
#     if not request.is_secure and os.getenv('FLASK_ENV') == 'production':
#         url = request.url.replace('http://', 'https://', 1)
#         return redirect(url, code=301)


# # ERROR HANDLERS
# @app.errorhandler(404)
# def not_found_error(error):
#     if request.path.startswith('/api/'):
#         return jsonify({'error': 'Not found'}), 404
#     return render_template('errors/404.html'), 404
#
# @app.errorhandler(500)
# def internal_error(error):
#     db.session.rollback()
#     app.logger.error(f'Server Error: {error}')
#     if request.path.startswith('/api/'):
#         return jsonify({'error': 'Internal server error'}), 500
#     return render_template('errors/500.html'), 500
#
# @app.errorhandler(429)
# def ratelimit_handler(e):
#     return jsonify({'error': 'Rate limit exceeded'}), 429


@app.template_filter('from_json')
def from_json_filter(s):
    """Safely parse JSON in templates"""
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_user_folder(user_id):
    """Get user data folder: data/users/user_{user_id}"""
    return os.path.join(app.config['USER_DATA_FOLDER'], f'user_{user_id}')

def get_chatbot_folder(user_id, chatbot_id):
    """Get chatbot folder: data/users/user_{user_id}/chatbots/chatbot_{chatbot_id}"""
    return os.path.join(get_user_folder(user_id), 'chatbots', f'chatbot_{chatbot_id}')

def get_avatar_path(user_id, chatbot_id, filename):
    """Get avatar file path inside chatbot folder"""
    return os.path.join(get_chatbot_folder(user_id, chatbot_id), filename)

def ensure_user_folder(user_id):
    """Ensure user folder exists"""
    user_folder = get_user_folder(user_id)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def ensure_chatbot_folder(user_id, chatbot_id):
    """Ensure chatbot folder exists"""
    chatbot_folder = get_chatbot_folder(user_id, chatbot_id)
    os.makedirs(chatbot_folder, exist_ok=True)
    return chatbot_folder


def subscription_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))

        user = db.session.get(User, session['user_id'])

        if not user:
            session.clear()
            flash('Please log in again', 'error')
            return redirect(url_for('login'))

        if not user.subscription:
            flash('You need an active subscription', 'warning')
            return redirect(url_for('subscription_plans'))

        if user.subscription.is_expired():
            flash('Your subscription has expired', 'error')
            return redirect(url_for('subscription_plans'))

        return f(*args, **kwargs)

    return decorated_function

@app.before_request
def setup_once():
    """Initialize on first request"""
    if not hasattr(app, 'initialized'):
        with app.app_context():
            initialize_subscription_plans()
        app.initialized = True


def calculate_similarity(str1, str2):
    """Calculate how similar two strings are (0.0 to 1.0)"""
    str1 = str1.strip().lower()
    str2 = str2.strip().lower()

    if str1 == str2:
        return 1.0

    if not str1 or not str2:
        return 0.0

    matcher = difflib.SequenceMatcher(None, str1, str2)
    return matcher.ratio()


def convert_urls_to_links(text):
    """
    Convert URLs in text to clickable HTML links
    Handles: http://, https://, and www. URLs
    """
    if not text or not isinstance(text, str):
        return text

    # URL regex pattern - matches http://, https://, and www. URLs
    url_pattern = r'(https?://[^\s]+)|(www\.[^\s]+)'

    def replace_url(match):
        url = match.group(0)
        href = url

        # Add https:// if URL starts with www.
        if url.startswith('www.'):
            href = 'https://' + url

        # Remove trailing punctuation if present
        trailing_punct = ''
        if url[-1] in '.,;:!?)':
            trailing_punct = url[-1]
            url = url[:-1]
            href = href[:-1]

        # Create clickable link with inline styles
        link = f'<a href="{href}" target="_blank" rel="noopener noreferrer" style="color: #4F46E5; text-decoration: underline; font-weight: 500;">{url}</a>{trailing_punct}'
        return link

    # Replace all URLs with clickable links
    converted_text = re.sub(url_pattern, replace_url, text)
    return converted_text


def detect_fallback_response(response: str, confidence: float) -> bool:
    """
    ✅ FIXED: Enhanced fallback detection with lower threshold
    """
    if not response:
        return True

    response_lower = response.lower()

    # ✅ Phrases that indicate VALID responses (NOT fallbacks)
    valid_response_indicators = [
        'hello', 'hi', 'hey', 'welcome', 'thank you', 'you\'re welcome',
        'goodbye', 'bye', 'see you', 'my name is', 'i am', 'i can help',
        'here\'s', 'here is', 'the answer', 'according to', 'based on',
        'doing great', 'functioning', 'here to help', 'glad', 'happy to'
    ]

    # If response contains valid indicators, it's NOT a fallback
    for indicator in valid_response_indicators:
        if indicator in response_lower:
            return False

    # ✅ FIXED: Lower threshold
    if confidence > 0.40:  # Lowered from 0.65
        return False

    # All possible fallback indicators from FallbackMessageManager
    fallback_indicators = [
        # Understanding issues
        "didn't fully understand",
        "may have missed",
        "could you clarify",
        "explain it differently",
        "could you please rephrase",
        "provide more context",
        "still learning",

        # Out of scope
        "don't have that information",
        "unable to help with that",
        "outside my current capabilities",
        "don't have access to that",
        "not in my current knowledge",

        # Retry suggestions
        "try asking in another way",
        "use simpler or more specific",
        "break your request into smaller",
        "let me know what outcome",
        "try a different approach",
        "try rephrasing",

        # Error handling
        "couldn't process that request",
        "something went wrong",
        "had trouble with that",
        "give it another shot",
        "try again",

        # Escalation
        "connect with a support representative",
        "contact our support team",
        "reach a human agent",
        "reaching out to our support",

        # Voice assistant
        "didn't catch that",
        "please repeat",
        "one more time",

        # Low confidence
        "not entirely confident",
        "need more information",
        "could you elaborate",

        # No match
        "understand your question, but don't have",
        "not in my current knowledge base",
        "that topic isn't covered",
        "don't have details about that",

        # Rate limiting
        "rate limit exceeded",

        # Configuration errors
        "configuration error",
        "chatbot id is required"
    ]

    # Check if response contains any fallback indicator
    for indicator in fallback_indicators:
            if indicator in response_lower:
                return True

    # Only mark as fallback if confidence is VERY low
            if confidence < 0.05:  # Changed from 0.10 to 0.05
                return True

    return False


def get_intent_response_from_json(user_message: str, training_data: dict) -> tuple:
    """Simple intent matcher from JSON"""
    if not training_data or not isinstance(training_data, dict):
        return None, 0.0, 'unknown'

    intents = training_data.get('intents', [])
    if not intents:
        return None, 0.0, 'unknown'

    best_match = None
    best_score = 0.0
    best_intent = 'unknown'
    threshold = 0.2

    user_message_clean = user_message.lower().strip()

    for intent in intents:
        patterns = intent.get('patterns', [])
        responses = intent.get('responses', [])
        tag = intent.get('tag', 'unknown')

        if not patterns or not responses:
            continue

        for pattern in patterns:
            score = calculate_similarity(user_message_clean, pattern)

            if score > best_score:
                best_score = score
                best_match = responses
                best_intent = tag

    if best_match and best_score >= threshold:
        response = random.choice(best_match)
        return response, best_score, best_intent

    return None, 0.0, 'unknown'


def get_bot_response(chatbot, user_message, context=None):
    """Bot response with chatbot isolation"""
    # ✅ LAZY IMPORT utils only when needed
    from utils import get_smart_response

    start_time = time.time()
    user_id = chatbot.user_id
    chatbot_id = chatbot.id
    session_id = context.get('session_id') if context else None

    print(f"\n{'=' * 70}")
    print(f"🤖 GET_BOT_RESPONSE (ISOLATED)")
    print(f"   User ID: {user_id}")
    print(f"   Chatbot ID: {chatbot_id}")
    print(f"   Session ID: {session_id}")
    print(f"   Message: {user_message[:50]}...")
    print(f"{'=' * 70}")

    try:
        response = get_smart_response(
            user_message,
            user_id,
            chatbot_id=chatbot_id,
            session_id=str(session_id)
        )

        if response and len(response) > 0:
            response = convert_urls_to_links(response)
            elapsed = time.time() - start_time
            print(f"  SUCCESS: Response generated (Elapsed: {elapsed:.2f}s)")
            return response

        print("\n  No valid response - using fallback")
        return "I'm here to help! Could you please rephrase your question?"

    except Exception as e:
        print(f"  CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "I encountered an error. Please try again."


def validate_intent_exists(intent_name: str, user_id: int, chatbot_id: int) -> bool:
    """
    Check if intent exists in knowledge base
    Returns True if intent is found, False otherwise
    """
    try:
        from utils import load_knowledge_base

        kb = load_knowledge_base(user_id, chatbot_id=chatbot_id)

        if not kb:
            return False

        intent_lower = intent_name.lower().strip()

        for item in kb:
            item_intent = (item.get('intent') or item.get('tag', '')).lower().strip()
            if item_intent == intent_lower:
                return True

        return False

    except Exception as e:
        print(f"⚠️ Intent validation error: {e}")
        return False


def process_button_action(action_type, action_value, chatbot):
    """
    ✅ FIXED: Enhanced button handler with proper intent detection

    Handles all button types:
    - url: Opens external link
    - intent: Looks up specific intent in KB
    - message: Uses smart response with intent detection
    - submenu: Shows submenu options
    """
    try:
        print(f"\n{'=' * 70}")
        print(f"🔘 BUTTON ACTION HANDLER")
        print(f"   Chatbot ID: {chatbot.id}")
        print(f"   Type: {action_type}")
        print(f"   Value: {action_value}")
        print(f"{'=' * 70}")

        # ============================================================
        # TYPE 1: URL - Open External Link
        # ============================================================
        if action_type == 'url':
            print(f"   🌐 URL Action")
            if action_value and action_value.strip():
                return f"🔗 Opening: {action_value}"
            return "❌ No URL provided."

        # ============================================================
        # TYPE 2: SUBMENU - Show Submenu Options
        # ============================================================
        elif action_type == 'submenu':
            print(f"   📂 Submenu Action")
            return "Please select an option from the menu above."

        # ============================================================
        # TYPE 3: INTENT - Direct Knowledge Base Lookup
        # ============================================================
        elif action_type == 'intent':
            print(f"   🎯 Intent Action: {action_value}")

            intent_name = action_value.lower().strip()

            # ✅ FIX 1: Validate intent exists first
            if not validate_intent_exists(intent_name, chatbot.user_id, chatbot.id):
                print(f"   ⚠️ Intent '{intent_name}' not found in KB")
                print(f"   → Falling back to smart response")

                # Fallback: Try smart response instead
                from utils import get_smart_response
                response = get_smart_response(
                    action_value,
                    chatbot.user_id,
                    chatbot_id=chatbot.id,
                    session_id=None
                )

                if response:
                    print(f"   ✓ Smart response found")
                    return response

                return f"I don't have information about {action_value.replace('_', ' ')}. Could you rephrase your question?"

            # Get response from KB
            from utils import get_response_from_kb
            response = get_response_from_kb(
                intent_name,
                chatbot.user_id,
                chatbot_id=chatbot.id
            )

            if response:
                print(f"   ✓ Intent response found")
                return response

            # Fallback if no response
            return f"I can help you with {action_value.replace('_', ' ')}. What would you like to know?"

        # ============================================================
        # TYPE 4: MESSAGE - Smart Response with Intent Detection
        # ============================================================
        elif action_type == 'message':
            print(f"   💬 Message Action: {action_value}")

            message_text = action_value.strip()

            if not message_text:
                return "No message provided."

            # ✅ FIX 2: Use smart response instead of literal text
            # This enables intent detection for message-type buttons
            from utils import get_smart_response

            print(f"   → Using smart response with intent detection")
            response = get_smart_response(
                message_text,
                chatbot.user_id,
                chatbot_id=chatbot.id,
                session_id=None
            )

            if response:
                print(f"   ✓ Smart response generated")
                return response

            # Fallback: Return literal text if no intent detected
            print(f"   → No intent detected, returning literal text")
            return message_text

        # ============================================================
        # DEFAULT: Unknown Action Type
        # ============================================================
        else:
            print(f"   ⚠️ Unknown action type: {action_type}")
            return "Please select an option from the menu."

    except Exception as e:
        print(f"❌ Button handler error: {e}")
        import traceback
        traceback.print_exc()
        return "I encountered an error. Please try selecting again."


# ============================================================
# VALIDATION HELPER FOR BUTTON CREATION/EDITING
# ============================================================

def validate_welcome_buttons(buttons_data, user_id, chatbot_id):
    """
    ✅ Validate welcome buttons before saving
    Checks if intent-type buttons have valid intents in KB

    Use this when creating/editing chatbots:

    validated_buttons = validate_welcome_buttons(
        buttons_list,
        user_id,
        chatbot_id
    )
    """
    if not isinstance(buttons_data, list):
        return buttons_data

    validated = []
    warnings = []

    for idx, button in enumerate(buttons_data):
        if not isinstance(button, dict):
            continue

        button_text = button.get('text', '').strip()
        button_type = button.get('type', 'url').strip()
        button_value = button.get('value', '').strip()
        has_submenu = button.get('has_submenu', False)
        submenu_items = button.get('submenu_items', [])

        if not button_text:
            continue

        # Validate main button
        if button_type == 'intent' and button_value:
            if not validate_intent_exists(button_value, user_id, chatbot_id):
                warning = f"⚠️ Button '{button_text}': Intent '{button_value}' not found in knowledge base"
                warnings.append(warning)
                print(warning)

        # Validate submenu items
        validated_submenu = []
        if has_submenu and isinstance(submenu_items, list):
            for sub_idx, sub_item in enumerate(submenu_items):
                if not isinstance(sub_item, dict):
                    continue

                sub_text = sub_item.get('text', '').strip()
                sub_type = sub_item.get('type', 'url').strip()
                sub_value = sub_item.get('value', '').strip()

                if not sub_text:
                    continue

                # Validate submenu intent
                if sub_type == 'intent' and sub_value:
                    if not validate_intent_exists(sub_value, user_id, chatbot_id):
                        warning = f" Submenu '{button_text} > {sub_text}': Intent '{sub_value}' not found"
                        warnings.append(warning)
                        print(warning)

                validated_submenu.append({
                    'text': sub_text,
                    'type': sub_type,
                    'value': sub_value
                })

        validated.append({
            'text': button_text,
            'type': button_type,
            'value': button_value,
            'has_submenu': has_submenu,
            'submenu_items': validated_submenu
        })

    # Print summary
    if warnings:
        print(f"\n{'=' * 70}")
        print(f"⚠️ BUTTON VALIDATION WARNINGS ({len(warnings)} issues)")
        print(f"{'=' * 70}")
        for warning in warnings:
            print(f"  {warning}")
        print(f"{'=' * 70}\n")

    return validated


def get_chatbot_response(user_message, user_id, session_id=None, chatbot_id=None):
    """Helper function with chatbot isolation"""
    try:
        print(f"\n  get_chatbot_response called")
        print(f"   User ID: {user_id}")
        print(f"   Chatbot ID: {chatbot_id}")

        response = get_smart_response(
            user_message,
            user_id,
            chatbot_id=chatbot_id,
            session_id=session_id
        )

        print(f"  Response: {response[:100]}...")
        return response
    except Exception as e:
        print(f" Error: {e}")

        if OLLAMA_AVAILABLE:
            try:
                ollama_response = generate_ollama_response(
                    user_message,
                    user_id,
                    session_id=session_id
                )
                if ollama_response:
                    return ollama_response
            except:
                pass

        return "I'm having trouble right now. Please try again."


def load_user_data(user_id):
    """Load user data and verify model exists"""
    user_folder = get_user_folder(user_id)
    model_path = os.path.join(user_folder, 'chatbot_model.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    intents_path = os.path.join(user_folder, 'intents.json')
    if not os.path.exists(intents_path):
        raise FileNotFoundError(f"Intents file not found at {intents_path}")

    return True


def validate_and_convert_training_data(data):
    """Validate training data format"""
    if not isinstance(data, dict):
        return None

    if 'intents' in data:
        intents = data['intents']
        if isinstance(intents, list) and len(intents) > 0:
            for intent in intents:
                if not all(k in intent for k in ['tag', 'patterns', 'responses']):
                    return None
            return data

    if 'training_data' in data:
        training_data = data['training_data']
        if isinstance(training_data, list):
            intents = []
            for idx, item in enumerate(training_data):
                if 'question' in item and 'answer' in item:
                    intents.append({
                        'tag': f'qa_{idx + 1}',
                        'patterns': [item['question']],
                        'responses': [item['answer']]
                    })
            if intents:
                return {'intents': intents}

    if isinstance(data, list):
        intents = []
        for idx, item in enumerate(data):
            if isinstance(item, dict) and 'question' in item and 'answer' in item:
                intents.append({
                    'tag': f'qa_{idx + 1}',
                    'patterns': [item['question']],
                    'responses': [item['answer']]
                })
        if intents:
            return {'intents': intents}

    return None


def parse_text_content(content):
    """Enhanced parser for text/HTML content to extract Q&A pairs"""
    import re
    from html.parser import HTMLParser

    class MLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reset()
            self.strict = False
            self.convert_charrefs = True
            self.text = []

        def handle_data(self, d):
            self.text.append(d)

        def get_data(self):
            return ''.join(self.text)

    stripper = MLStripper()
    try:
        stripper.feed(content)
        text = stripper.get_data()
    except:
        text = content

    qa_pairs = []

    qa_pattern = re.compile(
        r'(?:^|\n)\s*Q(?:uestion)?[:.\s]+(.+?)\s*(?:\n|$)\s*A(?:nswer)?[:.\s]+(.+?)(?=\n\s*Q(?:uestion)?[:.\s]|$)',
        re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    matches = qa_pattern.findall(text)
    if matches:
        for q, a in matches:
            q_clean = re.sub(r'\s+', ' ', q.strip())
            a_clean = re.sub(r'\s+', ' ', a.strip())
            if len(q_clean) > 3 and len(a_clean) > 3:
                qa_pairs.append({'question': q_clean, 'answer': a_clean})
        if qa_pairs:
            return qa_pairs

    numbered_pattern = re.compile(
        r'\d+\.\s*Q(?:uestion)?[:.\s]+(.+?)\s*A(?:nswer)?[:.\s]+(.+?)(?=\d+\.\s*Q(?:uestion)?[:.\s]|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = numbered_pattern.findall(text)
    if matches:
        for q, a in matches:
            q_clean = re.sub(r'\s+', ' ', q.strip())
            a_clean = re.sub(r'\s+', ' ', a.strip())
            if len(q_clean) > 3 and len(a_clean) > 3:
                qa_pairs.append({'question': q_clean, 'answer': a_clean})
        if qa_pairs:
            return qa_pairs

    formal_pattern = re.compile(
        r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=Question:|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = formal_pattern.findall(text)
    if matches:
        for q, a in matches:
            q_clean = re.sub(r'\s+', ' ', q.strip())
            a_clean = re.sub(r'\s+', ' ', a.strip())
            if len(q_clean) > 3 and len(a_clean) > 3:
                qa_pairs.append({'question': q_clean, 'answer': a_clean})
        if qa_pairs:
            return qa_pairs

    faq_pattern = re.compile(
        r'(?:^|\n)([^?\n]+\?)\s*(?:\n|$)\s*([^?\n]+?)(?=\n[^?\n]+\?|$)',
        re.MULTILINE | re.DOTALL
    )
    matches = faq_pattern.findall(text)
    if matches:
        for q, a in matches:
            q_clean = re.sub(r'\s+', ' ', q.strip())
            a_clean = re.sub(r'\s+', ' ', a.strip())
            if len(q_clean) > 10 and len(a_clean) > 10 and not a_clean.endswith('?'):
                qa_pairs.append({'question': q_clean, 'answer': a_clean})
        if qa_pairs:
            return qa_pairs

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    clean_lines = []
    for line in lines:
        if (len(line) < 5 or
                line.lower().startswith(('http', 'www', '©', 'copyright', 'all rights')) or
                line.lower() in ['faq', 'q&a', 'questions', 'answers']):
            continue

        line = re.sub(r'^\d+[.)]\s*', '', line)

        if len(line) > 5:
            clean_lines.append(line)

    for i in range(0, len(clean_lines) - 1, 2):
        question = clean_lines[i]
        answer = clean_lines[i + 1] if i + 1 < len(clean_lines) else ''

        if (question and answer and
                10 <= len(question) <= 300 and
                10 <= len(answer) <= 1000 and
                question.lower() != answer.lower()):
            qa_pairs.append({'question': question, 'answer': answer})

    seen = set()
    unique_pairs = []
    for pair in qa_pairs:
        key = (pair['question'].lower(), pair['answer'].lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    return unique_pairs


def get_existing_training_data(chatbot_id):
    """Get existing training data from database"""
    try:
        chatbot = db.session.get(Chatbot, chatbot_id)
        if not chatbot or not chatbot.training_data:
            return None
        return json.loads(chatbot.training_data)
    except Exception as e:
        print(f"Error loading existing training data: {e}")
        return None


def get_max_tag_number(intents):
    """Get maximum tag number from existing intents"""
    max_num = 0
    for intent in intents:
        tag = intent.get('tag', '')
        if tag.startswith('qa_'):
            try:
                num = int(tag.split('_')[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
    return max_num


def get_avatar_url(chatbot, request):
    """
    ✅ FIXED: Single source of truth for avatar URLs

    This ensures preview and embed use the same path logic
    """
    if not chatbot.bot_avatar:
        return None

    avatar_path = chatbot.bot_avatar.strip()
    base_url = request.url_root.rstrip('/')

    print(f"\n{'=' * 60}")
    print(f"🖼️ AVATAR URL GENERATION")
    print(f"   Chatbot: {chatbot.name} (ID: {chatbot.id})")
    print(f"   Raw Path: {avatar_path}")
    print(f"{'=' * 60}")

    # Case 1: Already a full URL
    if avatar_path.startswith(('http://', 'https://')):
        print(f"   ✓ Full URL: {avatar_path}")
        return avatar_path

    # Case 2: Path starts with /data/users/ (chatbot-specific)
    if avatar_path.startswith('/data/users/'):
        full_url = base_url + avatar_path
        print(f"   ✓ Chatbot folder: {full_url}")
        return full_url

    # Case 3: Path starts with /static/
    if avatar_path.startswith('/static/'):
        full_url = base_url + avatar_path
        print(f"   ✓ Static folder: {full_url}")
        return full_url

    # Case 4: Relative path without leading slash
    if not avatar_path.startswith('/'):
        avatar_path = f"/data/users/user_{chatbot.user_id}/chatbots/chatbot_{chatbot.id}/{avatar_path}"

    full_url = base_url + avatar_path
    print(f"   ✓ Constructed: {full_url}")

    # ✅ CRITICAL FIX: Verify file actually exists
    local_path = avatar_path.lstrip('/')
    if not os.path.exists(local_path):
        print(f"   ⚠️ WARNING: File not found at {local_path}")
        print(f"   → Avatar will not display")
        return None

    print(f"   ✅ File exists: {local_path}")
    print(f"{'=' * 60}\n")

    return full_url


# ============================================================================
# ISSUE 2: FIXED save_avatar_from_base64() - Robust Avatar Saving
# ============================================================================

def save_avatar_from_base64(base64_data, user_id, chatbot_id, is_temp=False):
    """
    ✅ FIXED: Robust avatar saving with proper error handling

    Args:
        base64_data: Base64 encoded image data
        user_id: User ID
        chatbot_id: Chatbot ID (None if creating new chatbot)
        is_temp: If True, save to temp location first

    Returns:
        tuple: (success: bool, path: str, error: str)
    """
    try:
        print(f"\n{'=' * 60}")
        print(f"💾 SAVING AVATAR")
        print(f"   User: {user_id} | Chatbot: {chatbot_id}")
        print(f"   Temp: {is_temp}")
        print(f"{'=' * 60}")

        if not base64_data or not base64_data.startswith('data:image'):
            return False, None, "Invalid base64 data"

        # Parse base64
        try:
            header, data = base64_data.split(',', 1)
            image_data = base64.b64decode(data)
        except Exception as e:
            return False, None, f"Failed to decode base64: {e}"

        # Open and validate image
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGBA')

            # Validate image size (optional: resize if too large)
            max_size = (500, 500)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                print(f"   ✓ Resized image to {image.size}")

        except Exception as e:
            return False, None, f"Failed to process image: {e}"

        # Determine save location
        if is_temp or chatbot_id is None:
            # Save to temp folder
            temp_folder = os.path.join('data', 'temp')
            os.makedirs(temp_folder, exist_ok=True)

            timestamp = str(int(time.time() * 1000))
            filename = f"avatar_temp_{user_id}_{timestamp}.png"
            filepath = os.path.join(temp_folder, filename)

            image.save(filepath, 'PNG')
            print(f"   ✓ Saved to temp: {filepath}")

            # Return temp marker
            return True, f"temp:{filename}", None

        else:
            # Save to chatbot folder
            chatbot_folder = os.path.join(
                'data', 'users', f'user_{user_id}',
                'chatbots', f'chatbot_{chatbot_id}'
            )
            os.makedirs(chatbot_folder, exist_ok=True)

            filename = "avatar.png"
            filepath = os.path.join(chatbot_folder, filename)

            # Delete old avatar if exists
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"   ✓ Deleted old avatar")
                except Exception as e:
                    print(f"   ⚠️ Could not delete old avatar: {e}")

            image.save(filepath, 'PNG')
            print(f"   ✓ Saved to chatbot folder: {filepath}")

            # Return database path
            db_path = f"/data/users/user_{user_id}/chatbots/chatbot_{chatbot_id}/{filename}"
            print(f"   ✓ Database path: {db_path}")
            print(f"{'=' * 60}\n")

            return True, db_path, None

    except Exception as e:
        print(f"   ❌ Avatar save error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, str(e)


# ============================================================================
# ISSUE 3: FIXED move_temp_avatar() - Move Temp Avatar After Chatbot Creation
# ============================================================================

def move_temp_avatar(temp_path, user_id, chatbot_id):
    """
    ✅ FIXED: Move temporary avatar to chatbot folder

    Args:
        temp_path: Temp path marker (e.g., "temp:avatar_temp_1_12345.png")
        user_id: User ID
        chatbot_id: Chatbot ID

    Returns:
        str: Final database path or None if failed
    """
    try:
        if not temp_path or not temp_path.startswith('temp:'):
            return None

        print(f"\n{'=' * 60}")
        print(f"📦 MOVING TEMP AVATAR")
        print(f"   User: {user_id} | Chatbot: {chatbot_id}")
        print(f"   Temp: {temp_path}")
        print(f"{'=' * 60}")

        # Extract filename
        temp_filename = temp_path.replace('temp:', '')
        temp_folder = os.path.join('data', 'temp')
        old_path = os.path.join(temp_folder, temp_filename)

        if not os.path.exists(old_path):
            print(f"   ❌ Temp file not found: {old_path}")
            return None

        # Create chatbot folder
        chatbot_folder = os.path.join(
            'data', 'users', f'user_{user_id}',
            'chatbots', f'chatbot_{chatbot_id}'
        )
        os.makedirs(chatbot_folder, exist_ok=True)

        # New path
        new_filename = "avatar.png"
        new_path = os.path.join(chatbot_folder, new_filename)

        # Move file
        shutil.move(old_path, new_path)
        print(f"   ✓ Moved: {old_path} → {new_path}")

        # Return database path
        db_path = f"/data/users/user_{user_id}/chatbots/chatbot_{chatbot_id}/{new_filename}"
        print(f"   ✓ Database path: {db_path}")
        print(f"{'=' * 60}\n")

        return db_path

    except Exception as e:
        print(f"   ❌ Move error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# ISSUE 4: FIXED delete_avatar() - Properly Delete Avatar Files
# ============================================================================

def delete_avatar(avatar_path):
    """
    ✅ FIXED: Delete avatar file from disk

    Args:
        avatar_path: Database path (e.g., "/data/users/user_1/chatbots/chatbot_1/avatar.png")

    Returns:
        bool: True if deleted successfully
    """
    try:
        if not avatar_path:
            return True

        print(f"\n{'=' * 60}")
        print(f"🗑️ DELETING AVATAR")
        print(f"   Path: {avatar_path}")
        print(f"{'=' * 60}")

        # Convert database path to file path
        if avatar_path.startswith('/data/'):
            file_path = avatar_path.lstrip('/')
        elif avatar_path.startswith('/static/'):
            file_path = avatar_path.lstrip('/')
        else:
            print(f"   ⚠️ Unknown path format: {avatar_path}")
            return False

        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   ✓ Deleted: {file_path}")
            print(f"{'=' * 60}\n")
            return True
        else:
            print(f"   ⚠️ File not found: {file_path}")
            print(f"{'=' * 60}\n")
            return False

    except Exception as e:
        print(f"   ❌ Delete error: {e}")
        print(f"{'=' * 60}\n")
        return False


def append_qa_pairs_to_db(chatbot_id, qa_data_list):
    """Append new QA pairs with duplicate prevention"""
    try:
        existing_pairs = QAPair.query.filter_by(chatbot_id=chatbot_id).all()

        existing_questions = set()
        max_tag_num = 0

        for pair in existing_pairs:
            existing_questions.add(pair.question.lower().strip())
            if pair.tag and pair.tag.startswith('qa_'):
                try:
                    num = int(pair.tag.split('_')[1])
                    max_tag_num = max(max_tag_num, num)
                except (ValueError, IndexError):
                    continue

        unique_qa_data = []
        for qa_data in qa_data_list:
            question_normalized = qa_data['question'].lower().strip()
            if question_normalized not in existing_questions:
                unique_qa_data.append(qa_data)
                existing_questions.add(question_normalized)

        if not unique_qa_data:
            print(f" All {len(qa_data_list)} Q&A pairs were duplicates")
            return True

        current_time = datetime.now(timezone.utc)
        for qa_data in unique_qa_data:
            max_tag_num += 1
            qa_pair = QAPair(
                chatbot_id=chatbot_id,
                question=qa_data['question'],
                answer=qa_data['answer'],
                tag=f'qa_{max_tag_num}',
                created_at=current_time,
                updated_at=current_time
            )
            db.session.add(qa_pair)

        db.session.commit()
        print(f" Appended {len(unique_qa_data)} unique Q&A pairs")
        return True

    except Exception as e:
        db.session.rollback()
        print(f"Error appending QA pairs: {e}")
        return False


def regenerate_intents_from_db(chatbot_id):
    """Regenerate intents and KB for specific chatbot only"""
    try:
        chatbot = Chatbot.query.get(chatbot_id)
        if not chatbot:
            return False

        qa_pairs = QAPair.query.filter_by(chatbot_id=chatbot_id).order_by(QAPair.created_at).all()

        intents = []
        for qa in qa_pairs:
            intents.append({
                'tag': qa.tag or f'qa_{qa.id}',
                'patterns': [qa.question],
                'responses': [qa.answer]
            })

        intents_data = {'intents': intents}
        chatbot.training_data = json.dumps(intents_data)

        chatbot_folder = ensure_chatbot_folder(chatbot.user_id, chatbot_id)

        intents_path = os.path.join(chatbot_folder, 'intents.json')
        with open(intents_path, 'w', encoding='utf-8') as f:
            json.dump(intents_data, f, indent=2)

        kb_path = os.path.join(chatbot_folder, 'knowledge_base.json')

        knowledge_base = []
        for intent in intents:
            knowledge_base.append({
                'intent': intent['tag'],
                'tag': intent['tag'],
                'patterns': intent['patterns'],
                'responses': intent['responses'],
                'chatbot_id': chatbot_id
            })

        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

        chatbot.intents_path = intents_path
        db.session.commit()

        clear_model_cache(chatbot.user_id, chatbot_id=chatbot_id)
        SMART_CACHE.clear_user(chatbot.user_id)

        print(f" Regenerated intents and KB for chatbot {chatbot_id}")
        return True

    except Exception as e:
        db.session.rollback()
        print(f" Error regenerating intents: {e}")
        import traceback
        traceback.print_exc()
        return False

# ROUTES
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


# Email validation
EMAIL_REGEX = re.compile(r"^[^@]+@[^@]+\.[^@]+$")


def domain_exists(email):
    """Check if the email domain actually exists"""
    try:
        domain = email.split('@')[-1]
        dns.resolver.resolve(domain, 'MX')
        return True
    except dns.resolver.NXDOMAIN:
        return False
    except Exception:
        return False


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration - Step 1"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        username = request.form.get('User_name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        if not all([first_name, last_name, username, email, phone, password, confirm_password]):
            flash('All fields are required.', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('register.html')

        if username == email:
            flash('Username and Email cannot be the same.', 'error')
            return render_template('register.html')

        if not EMAIL_REGEX.match(email):
            flash('Invalid email format.', 'error')
            return render_template('register.html')

        if not domain_exists(email):
            flash('Invalid email domain.', 'error')
            return render_template('register.html')

        if not phone.isdigit() or len(phone) != 10:
            flash('Phone number must be exactly 10 digits.', 'error')
            return render_template('register.html')

        try:
            hashed_password = generate_password_hash(password)
            new_user = User(
                first_name=first_name,
                last_name=last_name,
                username=username,
                email=email,
                phone=phone,
                password=hashed_password
            )

            db.session.add(new_user)
            db.session.commit()

            # Create user folder
            ensure_user_folder(new_user.id)

            # Store pending user info
            session['pending_user_id'] = new_user.id
            session['pending_email'] = new_user.email
            session['pending_username'] = new_user.username
            session['registration_success'] = True

            return redirect(url_for('subscription_plans'))

        except Exception as e:
            db.session.rollback()
            print(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'error')
            return render_template('register.html')

    return render_template('register.html')


#  SUBSCRIPTION PLANS ROUTE
@app.route('/subscription/plans')
def subscription_plans():
    """Show subscription plans page - Step 2"""
    # Initialize plans if needed
    if SubscriptionPlan.query.count() == 0:
        initialize_subscription_plans()

    plans = SubscriptionPlan.query.filter_by(is_active=True).all()

    current_plan = None
    is_pending_registration = False
    pending_username = None

    if 'pending_user_id' in session:
        is_pending_registration = True
        pending_username = session.get('pending_username', 'User')

        if session.pop('registration_success', False):
            flash('Account created successfully! Please choose a subscription plan to continue.', 'success')
    elif 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        if user and user.subscription:
            current_plan = user.subscription.plan.name

    return render_template(
        'subscription.html',
        plans=plans,
        current_plan=current_plan,
        is_pending_registration=is_pending_registration,
        pending_username=pending_username
    )


# SELECT SUBSCRIPTION ROUTE
@app.route('/subscription/select')
def select_subscription():
    """Handle plan selection - Step 3 (FIXED)"""
    plan_name = request.args.get('plan')

    if not plan_name:
        flash('Please select a plan.', 'error')
        return redirect(url_for('subscription_plans'))

    plan = SubscriptionPlan.query.filter_by(name=plan_name).first()
    if not plan:
        flash('Invalid plan selected.', 'error')
        return redirect(url_for('subscription_plans'))

    # Case 1: Newly registered user (pending)
    if 'pending_user_id' in session:
        user_id = session['pending_user_id']
        user = db.session.get(User, user_id)

        if not user:
            session.clear()
            flash('Registration error. Please try again.', 'error')
            return redirect(url_for('register'))

        try:
            # Check if user already has a subscription (prevent duplicate)
            existing_subscription = Subscription.query.filter_by(user_id=user_id).first()

            if existing_subscription:
                # Update existing subscription instead of creating new one
                existing_subscription.plan_id = plan.id
                existing_subscription.status = 'trial' if plan_name == 'free_trial' else 'active'
                existing_subscription.is_trial = (plan_name == 'free_trial')

                if plan_name == 'free_trial':
                    trial_end = datetime.now(timezone.utc) + timedelta(days=14)
                    existing_subscription.end_date = trial_end
                    existing_subscription.trial_end_date = trial_end
                else:
                    existing_subscription.end_date = datetime.now(timezone.utc) + timedelta(days=30)
                    existing_subscription.next_billing_date = existing_subscription.end_date

                db.session.commit()
                plan_message = f'Successfully subscribed to {plan.display_name} plan!'
            else:
                # Create new subscription
                if plan_name == 'free_trial':
                    subscription = create_trial_subscription(user_id)
                    if not subscription:
                        flash('Failed to activate trial. Please try again.', 'error')
                        return redirect(url_for('subscription_plans'))
                    plan_message = 'Your 14-day free trial has been activated!'
                else:
                    end_date = datetime.now(timezone.utc) + timedelta(days=30)
                    subscription = Subscription(
                        user_id=user_id,
                        plan_id=plan.id,
                        status='active',
                        is_trial=False,
                        end_date=end_date,
                        next_billing_date=end_date
                    )
                    db.session.add(subscription)
                    db.session.commit()
                    plan_message = f'Successfully subscribed to {plan.display_name} plan!'

            # Auto-login the user
            session.pop('pending_user_id', None)
            session.pop('pending_email', None)
            session.pop('pending_username', None)

            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email

            flash(plan_message, 'success')
            flash(f'Welcome to your dashboard, {user.first_name}!', 'info')
            return redirect(url_for('dashboard'))

        except Exception as e:
            db.session.rollback()
            print(f"Error creating subscription: {e}")
            import traceback
            traceback.print_exc()
            flash('Failed to activate subscription. Please try again.', 'error')
            return redirect(url_for('subscription_plans'))

    # Case 2: Logged-in user changing subscription
    elif 'user_id' in session:
        user_id = session['user_id']
        user = db.session.get(User, user_id)

        if not user:
            session.clear()
            flash('Session expired. Please log in again.', 'error')
            return redirect(url_for('login'))

        try:
            if user.subscription:
                # Update existing subscription
                old_plan = user.subscription.plan.display_name
                user.subscription.plan_id = plan.id
                user.subscription.status = 'trial' if plan_name == 'free_trial' else 'active'
                user.subscription.is_trial = (plan_name == 'free_trial')

                if plan_name == 'free_trial':
                    user.subscription.end_date = datetime.now(timezone.utc) + timedelta(days=14)
                    user.subscription.trial_end_date = user.subscription.end_date
                else:
                    user.subscription.end_date = datetime.now(timezone.utc) + timedelta(days=30)

                user.subscription.next_billing_date = user.subscription.end_date
                db.session.commit()
                flash(f'Successfully changed from {old_plan} to {plan.display_name}!', 'success')
            else:
                # Create new subscription
                if plan_name == 'free_trial':
                    subscription = create_trial_subscription(user_id)
                    if not subscription:
                        flash('Failed to activate trial. Please try again.', 'error')
                        return redirect(url_for('subscription_plans'))
                    flash('Your 14-day free trial has been activated!', 'success')
                else:
                    end_date = datetime.now(timezone.utc) + timedelta(days=30)
                    subscription = Subscription(
                        user_id=user_id,
                        plan_id=plan.id,
                        status='active',
                        is_trial=False,
                        end_date=end_date,
                        next_billing_date=end_date
                    )
                    db.session.add(subscription)
                    db.session.commit()
                    flash(f'Successfully subscribed to {plan.display_name} plan!', 'success')

            return redirect(url_for('dashboard'))

        except Exception as e:
            db.session.rollback()
            print(f"Error updating subscription: {e}")
            import traceback
            traceback.print_exc()
            flash('Failed to update subscription. Please try again.', 'error')
            return redirect(url_for('subscription_plans'))

    # Case 3: Not logged in or pending
    else:
        flash('Please register or log in to select a plan.', 'info')
        return redirect(url_for('register'))


# LOGIN ROUTE
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login with Remember Me functionality"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'  #Get remember checkbox

        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return render_template('login.html')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            #Check if user has subscription
            if not user.subscription:
                flash('Please select a subscription plan to access your account.', 'warning')
                session['pending_user_id'] = user.id
                session['pending_email'] = user.email
                session['pending_username'] = user.username
                return redirect(url_for('subscription_plans'))

            #Check subscription status
            if user.subscription.status in ['expired', 'cancelled']:
                session['user_id'] = user.id
                session['username'] = user.username
                session['email'] = user.email
                flash('Your subscription has expired. Please renew to continue.', 'warning')
                return redirect(url_for('subscription_plans'))

            #Set session with remember me
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email

            #Configure session permanence
            if remember:
                session.permanent = remember
                app.permanent_session_lifetime = timedelta(days=30)  # Remember for 30 days
            else:
                session.permanent = False
                # Session will expire when browser closes

            flash(f'Welcome back, {user.first_name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
            return render_template('login.html')

    return render_template('login.html')


#LOGOUT ROUTE
@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))


#SUBSCRIPTION MANAGEMENT ROUTES
@app.route('/subscription/manage')
@subscription_required
def manage_subscription():
    """Show subscription management page"""
    user = db.session.get(User, session['user_id'])
    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()

    # Get usage stats
    usage_stats = user.subscription.get_usage_stats()

    return render_template(
        'manage_subscription.html',
        user=user,
        subscription=user.subscription,
        chatbots=chatbots,
        usage_stats=usage_stats
    )


"""@app.route('/subscription/cancel', methods=['POST', 'GET'])
@subscription_required
def cancel_subscription():
    #Cancel user subscription
    user = User.query.get(session['user_id'])

    try:
        user.subscription.status = 'cancelled'
        db.session.commit()
        flash('Your subscription has been cancelled.', 'info')
    except Exception as e:
        db.session.rollback()
        print(f"Cancel error: {e}")
        flash('Failed to cancel subscription. Please try again.', 'error')

    return redirect(url_for('manage_subscription'))"""


@app.route('/subscription/reactivate')
@subscription_required
def reactivate_subscription():
    """Reactivate cancelled subscription"""
    user = db.session.get(User, session['user_id'])

    if not user.subscription or user.subscription.status != 'cancelled':
        flash('No cancelled subscription found.', 'error')
        return redirect(url_for('dashboard'))

    try:
        user.subscription.status = 'active'

        if user.subscription.end_date < datetime.now(timezone.utc):
            user.subscription.end_date = datetime.now(timezone.utc) + timedelta(days=30)
            user.subscription.next_billing_date = user.subscription.end_date

        db.session.commit()
        flash('Your subscription has been reactivated!', 'success')
    except Exception as e:
        db.session.rollback()
        print(f"Reactivate error: {e}")
        flash('Failed to reactivate subscription. Please try again.', 'error')

    return redirect(url_for('manage_subscription'))


@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    """Edit user profile"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])

    if request.method == 'POST':
        new_first_name = request.form.get('first_name')
        new_last_name = request.form.get('last_name')
        new_username = request.form.get('username')
        new_email = request.form.get('email')
        new_phone = request.form.get('phone')

        #Validation
        if not all([new_first_name, new_last_name, new_username, new_email, new_phone]):
            flash('All fields are required.', 'error')
            return redirect(url_for('edit_profile'))

        #Check if username is taken by another user
        if new_username != user.username:
            existing_user = User.query.filter_by(username=new_username).first()
            if existing_user:
                flash('Username already taken', 'error')
                return redirect(url_for('edit_profile'))

        #Check if email is taken by another user
        if new_email != user.email:
            existing_email = User.query.filter_by(email=new_email).first()
            if existing_email:
                flash('Email already registered', 'error')
                return redirect(url_for('edit_profile'))

            #Validate email format
            if not EMAIL_REGEX.match(new_email):
                flash('Invalid email format', 'error')
                return redirect(url_for('edit_profile'))

            if not domain_exists(new_email):
                flash('Invalid email domain', 'error')
                return redirect(url_for('edit_profile'))

        #Check if username and email are the same
        if new_username == new_email:
            flash('Username and Email cannot be the same.', 'error')
            return redirect(url_for('edit_profile'))

        #Validate phone number
        if not new_phone.isdigit() or len(new_phone) != 10:
            flash('Phone number must be exactly 10 digits.', 'error')
            return redirect(url_for('edit_profile'))

        #Update user information
        user.first_name = new_first_name
        user.last_name = new_last_name
        user.username = new_username
        user.email = new_email
        user.phone = new_phone

        db.session.commit()

        #Update session
        session['username'] = new_username
        session['email'] = new_email

        flash('Profile updated successfully!', 'success')
        return redirect(url_for('dashboard'))

    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()
    return render_template('edit_profile.html', user=user, chatbots=chatbots)



@app.route('/profile/change-password', methods=['GET', 'POST'])
def change_password():
    """Change user password"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        #Verify current password
        if not check_password_hash(user.password, current_password):
            flash('Current password is incorrect', 'error')
            return redirect(url_for('change_password'))

        #Check if new passwords match
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
            return redirect(url_for('change_password'))

        #Check password length
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return redirect(url_for('change_password'))

        #Update password
        user.password = generate_password_hash(new_password)
        db.session.commit()

        flash('Password changed successfully!', 'success')
        return redirect(url_for('dashboard'))

    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()
    return render_template('change_password.html', user=user, chatbots=chatbots)


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password request"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()

        if user:
            #Generate password reset token
            token = serializer.dumps(email, salt='password-reset-salt')

            #Create reset link
            reset_url = url_for('reset_password', token=token, _external=True)

            #Send email
            try:
                msg = Message(
                    'Password Reset Request',
                    recipients=[email]
                )
                msg.body = f'''Hello {user.username},

You have requested to reset your password. Click the link below to reset your password:

{reset_url}

This link will expire in 1 hour.

If you did not request this, please ignore this email.

Best regards,
ChatBot Builder Team
'''
                msg.html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            color: #333;
                        }}
                        .container {{
                            max-width: 600px;
                            margin: 0 auto;
                            padding: 20px;
                        }}
                        .header {{
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 30px;
                            text-align: center;
                            border-radius: 10px 10px 0 0;
                        }}
                        .content {{
                            background: #f7fafc;
                            padding: 30px;
                            border-radius: 0 0 10px 10px;
                        }}
                        .footer {{
                            text-align: center;
                            margin-top: 20px;
                            color: #718096;
                            font-size: 12px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">
                            <h1>Password Reset Request</h1>
                        </div>
                        <div class="content">
                            <p>Hello <strong>{user.username}</strong>,</p>
                            <p>You have requested to reset your password. Click the link below to reset your password:</p>

                            <p style="margin: 20px 0; text-align: center;">
                                <a href="{reset_url}" style="color: #667eea; text-decoration: underline; font-size: 16px; font-weight: 600;">Reset Password</a>
                            </p>

                            <p>Or copy and paste this link in your browser:</p>
                            <p style="word-break: break-all; color: #667eea;">
                                <a href="{reset_url}" style="color: #667eea;">{reset_url}</a>
                            </p>

                            <p><strong>This link will expire in 1 hour.</strong></p>
                            <p>If you did not request this, please ignore this email.</p>
                            <div class="footer">
                                <p>Best regards,<br>ChatBot Builder Team</p>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
                '''
                mail.send(msg)
                flash('Password reset link has been sent to your email', 'success')
            except Exception as e:
                print(f"Error sending email: {e}")
                flash('Error sending email. Please try again later.', 'error')
        else:
            # Don't reveal if email exists or not for security
            flash('If that email exists, a password reset link has been sent', 'info')

        return redirect(url_for('login'))

    return render_template('forgot_password.html')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Reset password with token"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    try:
        # Verify token (expires in 1 hour)
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except SignatureExpired:
        flash('The password reset link has expired', 'error')
        return redirect(url_for('forgot_password'))
    except BadSignature:
        flash('Invalid password reset link', 'error')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('reset_password', token=token))

        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return redirect(url_for('reset_password', token=token))

        # Update password
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(password)
            db.session.commit()
            flash('Password has been reset successfully! You can now login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found', 'error')
            return redirect(url_for('forgot_password'))

    return render_template('reset_password.html', token=token)


@app.route('/dashboard')
@subscription_required
def dashboard():
    """User dashboard - shows chatbots and subscription info"""
    user = db.session.get(User, session['user_id'])

    # Get all chatbots for this user
    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()

    # Calculate statistics
    total_chatbots = len(chatbots)
    active_chatbots = sum(1 for bot in chatbots if bot.is_active)

    # Check if ML model is trained
    user_folder = os.path.join(app.config['USER_DATA_FOLDER'], f'user_{user.id}')
    ml_model_trained = os.path.exists(os.path.join(user_folder, 'chatbot_model.h5'))

    # Subscription warnings
    subscription_warning = None
    if user.subscription:
        if user.subscription.is_trial and user.subscription.days_remaining() <= 3:
            subscription_warning = f"Your trial expires in {user.subscription.days_remaining()} days!"
        elif user.subscription.status == 'cancelled':
            subscription_warning = f"Your subscription is cancelled and will end in {user.subscription.days_remaining()} days."

    return render_template(
        'dashboard.html',
        user=user,
        chatbots=chatbots,
        total_chatbots=total_chatbots,
        active_chatbots=active_chatbots,
        ml_model_trained=ml_model_trained,
        subscription_warning=subscription_warning
    )


#CHATBOT CREATE ROUTE
@app.route('/chatbot/create', methods=['GET', 'POST'])
@subscription_required
def create_chatbot():
    """
    FIXED: Create route with proper submenu handling
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])

    #Check subscription limits
    if user.subscription and not user.subscription.can_create_chatbot():
        max_limit = user.subscription.plan.max_chatbots
        limit_text = "unlimited" if max_limit == -1 else str(max_limit)
        flash(f'You have reached your chatbot limit ({limit_text}). Please upgrade your plan.', 'error')
        return redirect(url_for('subscription_plans'))

    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        welcome_message = request.form.get('welcome_message', 'Hello! How can I help you?')
        theme_color = request.form.get('theme_color', '#4F46E5')
        bot_name = request.form.get('bot_name', 'AI Assistant')
        use_ml_model = request.form.get('use_ml_model') == 'on'

        #Colors
        chat_background_color = request.form.get('chat_background_color', '#F7FAFC')
        user_message_color = request.form.get('user_message_color', theme_color)
        bot_message_color = request.form.get('bot_message_color', '#FFFFFF')
        user_text_color = request.form.get('user_text_color', '#FFFFFF')
        bot_text_color = request.form.get('bot_text_color', '#1A202C')

        #Welcome Buttons with Submenu Support
        welcome_buttons_json = request.form.get('welcome_buttons', '[]')

        print(f"\n{'=' * 60}")
        print(f" CREATE CHATBOT - Processing Welcome Buttons")
        print(f"Raw data length: {len(welcome_buttons_json)}")
        print(f"{'=' * 60}")

        try:
            buttons_list = json.loads(welcome_buttons_json) if welcome_buttons_json else []

            if not isinstance(buttons_list, list):
                buttons_list = []

            #Validate and structure buttons with submenu support
            validated_buttons = []
            valid_types = ['url', 'intent', 'message', 'submenu']

            for idx, button in enumerate(buttons_list):
                if not isinstance(button, dict):
                    continue

                button_text = button.get('text', '').strip()
                button_type = button.get('type', 'url').strip()
                button_value = button.get('value', '').strip()
                has_submenu = button.get('has_submenu', False)
                submenu_items = button.get('submenu_items', [])

                if not button_text:
                    continue

                if button_type not in valid_types:
                    button_type = 'url'

                validated_button = {
                    'text': button_text,
                    'type': button_type,
                    'value': button_value,
                    'has_submenu': has_submenu
                }

                #Validate submenu items
                if has_submenu and isinstance(submenu_items, list):
                    validated_submenu = []
                    for sub_item in submenu_items:
                        if isinstance(sub_item, dict):
                            sub_text = sub_item.get('text', '').strip()
                            sub_type = sub_item.get('type', 'url').strip()
                            sub_value = sub_item.get('value', '').strip()

                            if sub_text:
                                if sub_type not in valid_types:
                                    sub_type = 'url'

                                validated_submenu.append({
                                    'text': sub_text,
                                    'type': sub_type,
                                    'value': sub_value
                                })

                    validated_button['submenu_items'] = validated_submenu
                    print(f"    Button '{button_text}' with {len(validated_submenu)} submenu items")
                else:
                    validated_button['submenu_items'] = []
                    print(f"    Button '{button_text}' (no submenu)")

                validated_buttons.append(validated_button)

            welcome_buttons = json.dumps(validated_buttons)
            print(f"    Total validated: {len(validated_buttons)} buttons")

        except json.JSONDecodeError as e:
            print(f"    JSON error: {e}")
            welcome_buttons = '[]'
        except Exception as e:
            print(f"    Validation error: {e}")
            welcome_buttons = '[]'

        #Avatar handling
        bot_avatar = None
        bot_avatar_data = request.form.get('bot_avatar_data', '').strip()

        if bot_avatar_data and bot_avatar_data.startswith('data:image'):
            try:
                import base64
                import io
                from PIL import Image
                import time

                header, data = bot_avatar_data.split(',', 1)
                image_data = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_data))
                image = image.convert('RGBA')

                timestamp = str(int(time.time()))
                # Save to temporary location first (chatbot not created yet)
                temp_folder = app.config['UPLOADS_FOLDER']
                os.makedirs(temp_folder, exist_ok=True)
                unique_filename = f"avatar_temp_{timestamp}.png"
                temp_path = os.path.join(temp_folder, unique_filename)
                image.save(temp_path, 'PNG')

                bot_avatar = f"temp_{unique_filename}"  # Mark as temporary

            except Exception as e:
                print(f"Avatar processing error: {e}")
                flash('Error processing avatar image', 'error')

        elif 'bot_avatar' in request.files:
            file = request.files['bot_avatar']
            if file and file.filename and allowed_file(file.filename):
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)

                if file_size > MAX_FILE_SIZE:
                    flash('Avatar file size must be less than 2MB', 'error')
                    return redirect(request.url)

                filename = secure_filename(f"{secrets.token_urlsafe(8)}_{file.filename}")
                filepath = os.path.join(app.config['AVATARS_FOLDER'], filename)
                file.save(filepath)
                bot_avatar = f'/static/uploads/avatars/{filename}'

        embed_code = secrets.token_urlsafe(16)

        new_chatbot = Chatbot(
            name=name,
            description=description,
            welcome_message=welcome_message,
            theme_color=theme_color,
            bot_name=bot_name,
            use_ml_model=use_ml_model,
            embed_code=embed_code,
            user_id=session['user_id'],
            bot_avatar=bot_avatar,
            chat_background_color=chat_background_color,
            user_message_color=user_message_color,
            bot_message_color=bot_message_color,
            user_text_color=user_text_color,
            bot_text_color=bot_text_color,
            welcome_buttons=welcome_buttons  #Store validated JSON
        )

        db.session.add(new_chatbot)

        if user.subscription:
            user.subscription.increment_chatbot_count()

        db.session.commit()

        #Move temporary avatar to chatbot folder
        if bot_avatar and bot_avatar.startswith('temp_'):
            try:
                chatbot_folder = ensure_chatbot_folder(new_chatbot.user_id, new_chatbot.id)
                old_filename = bot_avatar.replace('temp_', '')
                old_path = os.path.join(app.config['UPLOADS_FOLDER'], old_filename)
                new_filename = f"avatar.png"
                new_path = os.path.join(chatbot_folder, new_filename)

                if os.path.exists(old_path):
                    import shutil
                    shutil.move(old_path, new_path)
                    new_chatbot.bot_avatar = f"/data/users/user_{new_chatbot.user_id}/chatbots/chatbot_{new_chatbot.id}/{new_filename}"
                    db.session.commit()
            except Exception as e:
                print(f"Avatar move error: {e}")

        print(f"\n{'=' * 60}")
        print(f" CHATBOT CREATED")
        print(f"   ID: {new_chatbot.id}")
        print(f"   Buttons: {len(validated_buttons)} stored")
        print(f"{'=' * 60}\n")
        flash('Chatbot created successfully!', 'success')
        return redirect(url_for('dashboard'))

    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()
    return render_template('create_chatbot.html', chatbots=chatbots, user=user)


@app.route('/chatbot/edit/<int:chatbot_id>', methods=['GET', 'POST'])
def edit_chatbot(chatbot_id):
    """
      COMPLETELY FIXED: Edit chatbot with proper avatar handling
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))

    chatbot = db.session.get(Chatbot, chatbot_id)

    if not chatbot or chatbot.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))

    user = db.session.get(User, session['user_id'])
    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()

    if request.method == 'POST':
        try:
            #Basic fields
            chatbot.name = request.form.get('name', '').strip()
            chatbot.bot_name = request.form.get('bot_name', '').strip()
            chatbot.description = request.form.get('description', '').strip()
            chatbot.welcome_message = request.form.get('welcome_message', '').strip()

            #Colors
            chatbot.theme_color = request.form.get('theme_color', '#4F46E5')
            chatbot.chat_background_color = request.form.get('chat_background_color', '#F7FAFC')
            chatbot.bot_message_color = request.form.get('bot_message_color', '#FFFFFF')
            chatbot.bot_text_color = request.form.get('bot_text_color', '#1A202C')
            chatbot.user_text_color = request.form.get('user_text_color', '#FFFFFF')

            #1: Handle Avatar Changes Properly
            remove_avatar_flag = request.form.get('remove_avatar', 'false')
            bot_avatar_data = request.form.get('bot_avatar_data', '').strip()

            print(f"\n{'=' * 60}")
            print(f" AVATAR HANDLING")
            print(f"   Remove flag: {remove_avatar_flag}")
            print(f"   Avatar data length: {len(bot_avatar_data)}")
            print(f"   Current avatar: {chatbot.bot_avatar[:50] if chatbot.bot_avatar else 'None'}...")
            print(f"{'=' * 60}")

            # Case 1: User wants to remove avatar
            if remove_avatar_flag == 'true':
                print("    Removing avatar")

                # Delete old avatar file if it exists
                if chatbot.bot_avatar and chatbot.bot_avatar.startswith('/data/users/'):
                    # Extract path after /data/
                    relative_path = chatbot.bot_avatar.replace('/data/', '')
                    old_path = os.path.join('data', relative_path)
                    if os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                            print(f"   ✓ Deleted old avatar: {old_path}")
                        except Exception as e:
                            print(f"   ⚠️ Could not delete old avatar: {e}")

                chatbot.bot_avatar = None

            # Case 2: User uploaded new avatar (base64 data)
            elif bot_avatar_data and bot_avatar_data.startswith('data:image'):
                print("    Processing new base64 avatar")

                try:
                    import base64
                    import io
                    from PIL import Image
                    import time

                    #Delete old avatar file if it exists
                    if chatbot.bot_avatar and chatbot.bot_avatar.startswith('/data/users/'):
                        relative_path = chatbot.bot_avatar.replace('/data/', '')
                        old_path = os.path.join('data', relative_path)
                        if os.path.exists(old_path):
                            try:
                                os.remove(old_path)
                                print(f"   ✓ Deleted old avatar: {old_path}")
                            except Exception as e:
                                print(f"   ⚠️ Could not delete old avatar: {e}")

                    #Process new avatar
                    header, data = bot_avatar_data.split(',', 1)
                    image_data = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_data))
                    image = image.convert('RGBA')

                    #Create unique filename
                    timestamp = str(int(time.time()))
                    # Save to chatbot-specific folder
                    chatbot_folder = ensure_chatbot_folder(chatbot.user_id, chatbot_id)
                    unique_filename = f"avatar.png"
                    file_path = os.path.join(chatbot_folder, unique_filename)
                    image.save(file_path, 'PNG')

                    #Update chatbot with new avatar path
                    chatbot.bot_avatar = f"/data/users/user_{chatbot.user_id}/chatbots/chatbot_{chatbot_id}/{unique_filename}"

                    print(f"   ✓ Saved new avatar: {chatbot.bot_avatar}")

                except Exception as e:
                    print(f"    Avatar processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    flash('Error processing avatar image', 'error')

            # Case 3: User uploaded file directly
            elif 'bot_avatar' in request.files:
                file = request.files['bot_avatar']
                if file and file.filename and allowed_file(file.filename):
                    print("   → Processing uploaded file")

                    # Check file size
                    file.seek(0, os.SEEK_END)
                    file_size = file.tell()
                    file.seek(0)

                    if file_size > MAX_FILE_SIZE:
                        flash('Avatar file size must be less than 2MB', 'error')
                        return redirect(request.url)

                    #Delete old avatar file if it exists
                    if chatbot.bot_avatar and chatbot.bot_avatar.startswith('/data/users/'):
                        relative_path = chatbot.bot_avatar.replace('/data/', '')
                        old_path = os.path.join('data', relative_path)
                        if os.path.exists(old_path):
                            try:
                                os.remove(old_path)
                                print(f"   ✓ Deleted old avatar: {old_path}")
                            except Exception as e:
                                print(f"    Could not delete old avatar: {e}")
                    #Save new file
                    chatbot_folder = ensure_chatbot_folder(chatbot.user_id, chatbot_id)
                    filename = f"avatar.png"
                    filepath = os.path.join(chatbot_folder, filename)
                    file.save(filepath)
                    chatbot.bot_avatar = f'/data/users/user_{chatbot.user_id}/chatbots/chatbot_{chatbot_id}/{filename}'

                    print(f"   ✓ Saved file: {chatbot.bot_avatar}")

            else:
                print("   → No avatar changes")

            #Welcome Buttons with Submenu Support
            welcome_buttons_raw = request.form.get('welcome_buttons', '[]')

            print(f"\n{'=' * 60}")
            print(f" PROCESSING WELCOME BUTTONS")
            print(f"   Raw data length: {len(welcome_buttons_raw)}")
            print(f"{'=' * 60}")

            try:
                if isinstance(welcome_buttons_raw, str):
                    buttons_list = json.loads(welcome_buttons_raw) if welcome_buttons_raw else []
                else:
                    buttons_list = welcome_buttons_raw if welcome_buttons_raw else []

                if not isinstance(buttons_list, list):
                    buttons_list = []

                #Validate and structure buttons
                validated_buttons = []
                valid_types = ['url', 'intent', 'message', 'submenu']

                for idx, button in enumerate(buttons_list):
                    if not isinstance(button, dict):
                        continue

                    button_text = button.get('text', '').strip()
                    button_type = button.get('type', 'url').strip()
                    button_value = button.get('value', '').strip()
                    has_submenu = button.get('has_submenu', False)
                    submenu_items = button.get('submenu_items', [])

                    if not button_text:
                        continue

                    if button_type not in valid_types:
                        button_type = 'url'

                    validated_button = {
                        'text': button_text,
                        'type': button_type,
                        'value': button_value,
                        'has_submenu': has_submenu
                    }

                    #Validate submenu items
                    if has_submenu and isinstance(submenu_items, list):
                        validated_submenu = []
                        for sub_item in submenu_items:
                            if isinstance(sub_item, dict):
                                sub_text = sub_item.get('text', '').strip()
                                sub_type = sub_item.get('type', 'url').strip()
                                sub_value = sub_item.get('value', '').strip()

                                if sub_text:
                                    if sub_type not in valid_types:
                                        sub_type = 'url'

                                    validated_submenu.append({
                                        'text': sub_text,
                                        'type': sub_type,
                                        'value': sub_value
                                    })

                        validated_button['submenu_items'] = validated_submenu
                        print(f"   Button '{button_text}' with {len(validated_submenu)} submenu items")
                    else:
                        validated_button['submenu_items'] = []
                        print(f"   Button '{button_text}' (no submenu)")

                    validated_buttons.append(validated_button)

                chatbot.welcome_buttons = json.dumps(validated_buttons)
                print(f"    Validated {len(validated_buttons)} buttons")

            except json.JSONDecodeError as e:
                print(f"    JSON decode error: {e}")
                chatbot.welcome_buttons = '[]'
            except Exception as e:
                print(f"    Validation error: {e}")
                chatbot.welcome_buttons = '[]'

            #Other settings
            chatbot.use_ml_model = request.form.get('use_ml_model') == 'on'

            from datetime import datetime, timezone
            chatbot.updated_at = datetime.now(timezone.utc)

            #Commit changes to database
            db.session.commit()

            print(f"\n CHATBOT UPDATED SUCCESSFULLY!")
            print(f"   Avatar: {chatbot.bot_avatar[:50] if chatbot.bot_avatar else 'None'}...")
            print(f"   Buttons: {len(validated_buttons)}")
            print(f"{'=' * 60}\n")

            flash('Chatbot updated successfully!', 'success')
            return redirect(url_for('dashboard'))

        except Exception as e:
            db.session.rollback()
            print(f" ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'Error updating chatbot: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('edit_chatbot.html', chatbot=chatbot, chatbots=chatbots, user=user)



#Training Route
@app.route('/chatbot/train/<int:chatbot_id>', methods=['GET', 'POST'])
def train_chatbot(chatbot_id):
    """✅ FIXED: Chatbot-specific training with complete isolation"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    chatbot = Chatbot.query.get_or_404(chatbot_id)

    if chatbot.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))

    user_id = session['user_id']

    #Use chatbot-specific folder, NOT user folder
    chatbot_folder = get_chatbot_folder(chatbot.user_id, chatbot_id)
    os.makedirs(chatbot_folder, exist_ok=True)

    if request.method == 'POST':
        upload_mode = request.form.get('upload_mode', 'append')

        # Handle JSON file upload
        if 'training_file' in request.files:
            file = request.files['training_file']

            if file and file.filename:
                if not file.filename.lower().endswith('.json'):
                    return jsonify({'success': False, 'message': 'Please upload a JSON file'})

                try:
                    file_content = file.read().decode('utf-8')
                    parsed_data = json.loads(file_content)
                    converted_data = validate_and_convert_training_data(parsed_data)

                    if not converted_data:
                        return jsonify({'success': False, 'message': 'Invalid training data format'})

                    # Handle append mode
                    if upload_mode == 'append':
                        existing_data = get_existing_training_data(chatbot_id)
                        if existing_data:
                            existing_intents = existing_data.get('intents', [])
                            new_intents = converted_data.get('intents', [])
                            max_tag_num = get_max_tag_number(existing_intents)

                            for intent in new_intents:
                                max_tag_num += 1
                                intent['tag'] = f'qa_{max_tag_num}'

                            merged_intents = existing_intents + new_intents
                            converted_data = {'intents': merged_intents}

                    # Save to database
                    qa_list = []
                    for intent in converted_data.get('intents', []):
                        if intent.get('patterns') and intent.get('responses'):
                            qa_list.append({
                                'question': intent['patterns'][0],
                                'answer': intent['responses'][0],
                                'tag': intent.get('tag', '')
                            })

                    if upload_mode == 'replace':
                        save_qa_pairs_to_db(chatbot_id, qa_list)
                    else:
                        append_qa_pairs_to_db(chatbot_id, qa_list)

                    # 1: Save to CHATBOT-SPECIFIC folder
                    intents_path = os.path.join(chatbot_folder, 'intents.json')
                    with open(intents_path, 'w', encoding='utf-8') as f:
                        json.dump(converted_data, f, indent=2)

                    # 2: Update chatbot record
                    chatbot.training_data = json.dumps(converted_data)
                    chatbot.intents_path = intents_path
                    chatbot.training_file = file.filename
                    chatbot.trained_folder = os.path.join(chatbot_folder, 'models')
                    chatbot.is_trained = True
                    db.session.commit()

                    # 3: Use chatbot-specific training
                    print(f"\n{'=' * 60}")
                    print(f"🤖 AUTO-TRAINING ML MODEL FOR CHATBOT {chatbot_id}")
                    print(f"{'=' * 60}\n")

                    try:
                        from model_training import train_chatbot_model
                        from utils import clear_model_cache

                        #Train chatbot-specific model
                        train_result = train_chatbot_model(user_id, chatbot_id)

                        #Clear chatbot-specific cache
                        clear_model_cache(user_id, chatbot_id=chatbot_id)

                        chatbot.use_ml_model = True
                        db.session.commit()

                        action = "appended to" if upload_mode == 'append' else "trained"
                        return jsonify({
                            'success': True,
                            'message': f' Successfully {action} {len(converted_data.get("intents", []))} intents with {train_result.get("accuracy", 0):.1%} accuracy!',
                            'intent_count': len(converted_data.get("intents", [])),
                            'ml_trained': True,
                            'chatbot_id': chatbot_id
                        })
                    except Exception as e:
                        print(f" ML training error: {e}")
                        import traceback
                        traceback.print_exc()

                        return jsonify({
                            'success': True,
                            'message': f' Training data saved, but ML training failed: {str(e)}',
                            'intent_count': len(converted_data.get("intents", [])),
                            'ml_trained': False
                        })

                except json.JSONDecodeError:
                    return jsonify({'success': False, 'message': 'Invalid JSON format'})
                except Exception as e:
                    print(f"File upload error: {e}")
                    import traceback
                    traceback.print_exc()
                    return jsonify({'success': False, 'message': f'Error: {str(e)}'})

        #Handle manual Q&A entry
        if request.is_json or request.form.get('training_data'):
            try:
                if request.is_json:
                    data = request.get_json()
                    training_data_str = data.get('training_data')
                    upload_mode = data.get('upload_mode', 'append')
                else:
                    training_data_str = request.form.get('training_data')
                    upload_mode = request.form.get('upload_mode', 'append')

                if not training_data_str:
                    return jsonify({'success': False, 'message': 'No training data provided'})

                parsed_data = json.loads(training_data_str)
                converted_data = validate_and_convert_training_data(parsed_data)

                if not converted_data:
                    return jsonify({'success': False, 'message': 'Invalid training data format'})

                # Handle append mode
                if upload_mode == 'append':
                    existing_data = get_existing_training_data(chatbot_id)
                    if existing_data:
                        existing_intents = existing_data.get('intents', [])
                        new_intents = converted_data.get('intents', [])
                        max_tag_num = get_max_tag_number(existing_intents)

                        for intent in new_intents:
                            max_tag_num += 1
                            intent['tag'] = f'qa_{max_tag_num}'

                        merged_intents = existing_intents + new_intents
                        converted_data = {'intents': merged_intents}

                #Save to database
                qa_list = []
                for intent in converted_data.get('intents', []):
                    if intent.get('patterns') and intent.get('responses'):
                        qa_list.append({
                            'question': intent['patterns'][0],
                            'answer': intent['responses'][0],
                            'tag': intent.get('tag', '')
                        })

                if upload_mode == 'replace':
                    save_qa_pairs_to_db(chatbot_id, qa_list)
                else:
                    append_qa_pairs_to_db(chatbot_id, qa_list)

                #Regenerate chatbot-specific intents
                regenerate_intents_from_db(chatbot_id)

                chatbot.training_file = 'manual_entry.json'
                chatbot.is_trained = True
                db.session.commit()

                #Train chatbot-specific model
                print(f"\n{'=' * 60}")
                print(f" AUTO-TRAINING ML MODEL FOR CHATBOT {chatbot_id}")
                print(f"{'=' * 60}\n")

                try:
                    from model_training import train_chatbot_model
                    from utils import clear_model_cache

                    train_result = train_chatbot_model(user_id, chatbot_id)
                    clear_model_cache(user_id, chatbot_id=chatbot_id)

                    chatbot.use_ml_model = True
                    db.session.commit()

                    action = "appended and trained" if upload_mode == 'append' else "trained"
                    return jsonify({
                        'success': True,
                        'message': f' Successfully {action} {len(converted_data.get("intents", []))} intents with {train_result.get("accuracy", 0):.1%} accuracy!',
                        'intent_count': len(converted_data.get("intents", [])),
                        'ml_trained': True,
                        'chatbot_id': chatbot_id
                    })
                except Exception as e:
                    print(f" ML training error: {e}")
                    import traceback
                    traceback.print_exc()

                    return jsonify({
                        'success': True,
                        'message': f' Training data saved, but ML training failed: {str(e)}',
                        'intent_count': len(converted_data.get("intents", [])),
                        'ml_trained': False
                    })

            except json.JSONDecodeError:
                return jsonify({'success': False, 'message': 'Invalid JSON format'})
            except Exception as e:
                print(f"Manual training error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'success': False, 'message': f'Error: {str(e)}'})

    #GET request
    user = db.session.get(User, session['user_id'])
    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()
    return render_template('train_chatbot.html', chatbot=chatbot, chatbots=chatbots, user=user)


@app.route('/chatbot/train-text/<int:chatbot_id>', methods=['POST'])
def train_chatbot_text(chatbot_id):
    """Handle text/HTML upload with chatbot isolation"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    chatbot = Chatbot.query.get_or_404(chatbot_id)

    if chatbot.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    if 'text_file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})

    file = request.files['text_file']
    upload_mode = request.form.get('upload_mode', 'append')

    if not file or not file.filename:
        return jsonify({'success': False, 'message': 'No file selected'})

    allowed_extensions = ['.txt', '.html', '.htm']
    file_ext = file.filename[file.filename.rfind('.'):].lower()

    if file_ext not in allowed_extensions:
        return jsonify({'success': False, 'message': 'Please upload a TXT or HTML file'})

    try:
        content = file.read().decode('utf-8', errors='ignore')
        qa_pairs = parse_text_content(content)

        if not qa_pairs:
            return jsonify({
                'success': False,
                'message': 'Could not extract Q&A pairs from file. Please check the format.'
            })

        #Handle append mode
        start_index = 1
        if upload_mode == 'append':
            existing_data = get_existing_training_data(chatbot_id)
            if existing_data:
                existing_intents = existing_data.get('intents', [])
                start_index = get_max_tag_number(existing_intents) + 1

        #Save to database
        qa_list = [
            {
                'question': qa['question'],
                'answer': qa['answer'],
                'tag': f'qa_{start_index + idx}'
            }
            for idx, qa in enumerate(qa_pairs)
        ]

        if upload_mode == 'replace':
            save_qa_pairs_to_db(chatbot_id, qa_list)
        else:
            append_qa_pairs_to_db(chatbot_id, qa_list)

        #Regenerate chatbot-specific intents
        regenerate_intents_from_db(chatbot_id)

        chatbot.is_trained = True
        db.session.commit()

        #Train chatbot-specific model
        print(f"\n{'=' * 60}")
        print(f" AUTO-TRAINING ML MODEL FOR CHATBOT {chatbot_id}")
        print(f"{'=' * 60}\n")

        try:
            from model_training import train_chatbot_model
            from utils import clear_model_cache

            train_result = train_chatbot_model(session['user_id'], chatbot_id)
            clear_model_cache(session['user_id'], chatbot_id=chatbot_id)

            chatbot.use_ml_model = True
            db.session.commit()

            action = "appended and trained" if upload_mode == 'append' else "extracted and trained"
            return jsonify({
                'success': True,
                'message': f' {action.capitalize()} {len(qa_pairs)} Q&A pairs with {train_result.get("accuracy", 0):.1%} accuracy!',
                'intent_count': len(qa_pairs),
                'ml_trained': True,
                'chatbot_id': chatbot_id
            })
        except Exception as e:
            print(f" ML training error: {e}")
            import traceback
            traceback.print_exc()

            return jsonify({
                'success': True,
                'message': f' Training data saved, but ML training failed: {str(e)}',
                'intent_count': len(qa_pairs),
                'ml_trained': False
            })

    except Exception as e:
        print(f"Error processing text file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})


@app.route('/chatbot/train-excel/<int:chatbot_id>', methods=['POST'])
def train_chatbot_excel(chatbot_id):
    """Handle Excel file upload with chatbot isolation"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    chatbot = Chatbot.query.get_or_404(chatbot_id)

    if chatbot.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    if 'excel_file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})

    file = request.files['excel_file']
    upload_mode = request.form.get('upload_mode', 'append')

    if not file or not file.filename:
        return jsonify({'success': False, 'message': 'No file selected'})

    allowed_extensions = ['.xlsx', '.xls', '.csv']
    file_ext = file.filename[file.filename.rfind('.'):].lower()

    if file_ext not in allowed_extensions:
        return jsonify({'success': False, 'message': 'Please upload an Excel or CSV file'})

    try:
        import pandas as pd
        from io import BytesIO

        # Read Excel file
        file_bytes = BytesIO(file.read())

        if file_ext == '.csv':
            df = pd.read_csv(file_bytes)
        else:
            df = pd.read_excel(file_bytes)

        # Validate columns
        if 'Question' not in df.columns or 'Answer' not in df.columns:
            return jsonify({
                'success': False,
                'message': 'Excel file must have "Question" and "Answer" columns'
            })

        # Extract Q&A pairs
        qa_pairs = []
        for _, row in df.iterrows():
            question = str(row['Question']).strip()
            answer = str(row['Answer']).strip()

            if question and answer and question != 'nan' and answer != 'nan':
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })

        if not qa_pairs:
            return jsonify({
                'success': False,
                'message': 'No valid Q&A pairs found in Excel file'
            })

        # Handle append mode
        start_index = 1
        if upload_mode == 'append':
            existing_data = get_existing_training_data(chatbot_id)
            if existing_data:
                existing_intents = existing_data.get('intents', [])
                start_index = get_max_tag_number(existing_intents) + 1

        # Save to database
        qa_list = [
            {
                'question': qa['question'],
                'answer': qa['answer'],
                'tag': f'qa_{start_index + idx}'
            }
            for idx, qa in enumerate(qa_pairs)
        ]

        if upload_mode == 'replace':
            save_qa_pairs_to_db(chatbot_id, qa_list)
        else:
            append_qa_pairs_to_db(chatbot_id, qa_list)

        # Regenerate chatbot-specific intents
        regenerate_intents_from_db(chatbot_id)

        chatbot.is_trained = True
        db.session.commit()

        # Train chatbot-specific model
        try:
            from model_training import train_chatbot_model
            from utils import clear_model_cache

            train_result = train_chatbot_model(session['user_id'], chatbot_id)
            clear_model_cache(session['user_id'], chatbot_id=chatbot_id)

            chatbot.use_ml_model = True
            db.session.commit()

            action = "imported and trained" if upload_mode == 'append' else "extracted and trained"
            return jsonify({
                'success': True,
                'message': f'✓ {action.capitalize()} {len(qa_pairs)} Q&A pairs from Excel with {train_result.get("accuracy", 0):.1%} accuracy!',
                'intent_count': len(qa_pairs),
                'ml_trained': True,
                'chatbot_id': chatbot_id
            })
        except Exception as e:
            print(f"ML training error: {e}")
            return jsonify({
                'success': True,
                'message': f'✓ Imported {len(qa_pairs)} Q&A pairs from Excel, but ML training failed',
                'intent_count': len(qa_pairs),
                'ml_trained': False
            })

    except Exception as e:
        print(f"Error processing Excel file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})


# ============================================================
# ADD THESE ROUTES TO YOUR app.py
# Place them after your existing /chatbot/train routes
# ============================================================

@app.route('/chatbot/train-manual/<int:chatbot_id>', methods=['POST'])
def train_chatbot_manual(chatbot_id):
    """Handle manual Q&A entry with chatbot isolation"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    chatbot = Chatbot.query.get_or_404(chatbot_id)

    if chatbot.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    try:
        data = request.get_json()
        qa_pairs = data.get('qa_pairs', [])
        upload_mode = data.get('upload_mode', 'append')

        if not qa_pairs:
            return jsonify({'success': False, 'message': 'No Q&A pairs provided'})

        print(f"\n{'=' * 60}")
        print(f"📝 MANUAL Q&A TRAINING")
        print(f"   Chatbot ID: {chatbot_id}")
        print(f"   Mode: {upload_mode}")
        print(f"   Q&A Pairs: {len(qa_pairs)}")
        print(f"{'=' * 60}\n")

        # Handle append mode - get starting index
        start_index = 1
        if upload_mode == 'append':
            existing_data = get_existing_training_data(chatbot_id)
            if existing_data:
                existing_intents = existing_data.get('intents', [])
                start_index = get_max_tag_number(existing_intents) + 1

        # Prepare Q&A list for database
        qa_list = []
        for idx, qa in enumerate(qa_pairs):
            question = qa.get('question', '').strip()
            answer = qa.get('answer', '').strip()
            qa_id = qa.get('id')  # Maybe None for new pairs

            if not question or not answer:
                continue

            # If editing existing pair
            if qa_id and isinstance(qa_id, int):
                existing_qa = QAPair.query.get(qa_id)
                if existing_qa and existing_qa.chatbot_id == chatbot_id:
                    existing_qa.question = question
                    existing_qa.answer = answer
                    existing_qa.updated_at = datetime.now(timezone.utc)
                    print(f"   ✏️  Updated Q&A #{qa_id}")
                    continue

            # New Q&A pair
            qa_list.append({
                'question': question,
                'answer': answer,
                'tag': f'qa_{start_index + idx}'
            })

        # Save new pairs to database
        if qa_list:
            if upload_mode == 'replace':
                save_qa_pairs_to_db(chatbot_id, qa_list)
                print(f"   ✅ Replaced with {len(qa_list)} new pairs")
            else:
                append_qa_pairs_to_db(chatbot_id, qa_list)
                print(f"   ✅ Appended {len(qa_list)} new pairs")

        # Commit any updates to existing pairs
        db.session.commit()

        # Regenerate chatbot-specific intents and KB
        regenerate_intents_from_db(chatbot_id)

        chatbot.is_trained = True
        db.session.commit()

        # Train chatbot-specific ML model
        try:
            from model_training import train_chatbot_model
            from utils import clear_model_cache

            train_result = train_chatbot_model(session['user_id'], chatbot_id)
            clear_model_cache(session['user_id'], chatbot_id=chatbot_id)

            chatbot.use_ml_model = True
            db.session.commit()

            total_pairs = len(qa_pairs)
            action = "saved and trained" if upload_mode == 'append' else "replaced and trained"

            return jsonify({
                'success': True,
                'message': f'✓ Successfully {action} {total_pairs} Q&A pairs with {train_result.get("accuracy", 0):.1%} accuracy!',
                'intent_count': total_pairs,
                'ml_trained': True,
                'chatbot_id': chatbot_id
            })

        except Exception as e:
            print(f"❌ ML training error: {e}")
            import traceback
            traceback.print_exc()

            return jsonify({
                'success': True,
                'message': f'✓ Q&A pairs saved, but ML training failed: {str(e)}',
                'intent_count': len(qa_pairs),
                'ml_trained': False
            })

    except Exception as e:
        db.session.rollback()
        print(f" Manual training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})



#Delete_qa_pair_route
@app.route('/chatbot/delete-qa-pair/<int:qa_id>', methods=['POST'])
def delete_qa_pair_route(qa_id):
    """ FIXED: Delete a single Q&A pair with proper cache clearing"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    try:
        from database import delete_qa_pair

        qa_pair = QAPair.query.get(qa_id)
        if not qa_pair:
            return jsonify({'success': False, 'message': 'Q&A pair not found'})

        # Check ownership
        chatbot = Chatbot.query.get(qa_pair.chatbot_id)
        if chatbot.user_id != session['user_id']:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403

        chatbot_id = qa_pair.chatbot_id
        user_id = chatbot.user_id

        if delete_qa_pair(qa_id):
            #  FIX 1: Regenerate intents.json
            regenerate_intents_from_db(chatbot_id)

            #  FIX 2: Regenerate knowledge_base.json
            user_folder = get_user_folder(user_id)
            kb_path = os.path.join(user_folder, 'knowledge_base.json')

            if chatbot.training_data:
                try:
                    intents_data = json.loads(chatbot.training_data)

                    knowledge_base = []
                    for intent in intents_data.get('intents', []):
                        knowledge_base.append({
                            'intent': intent.get('tag'),
                            'tag': intent.get('tag'),
                            'patterns': intent.get('patterns', []),
                            'responses': intent.get('responses', [])
                        })

                    os.makedirs(user_folder, exist_ok=True)
                    with open(kb_path, 'w', encoding='utf-8') as f:
                        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)

                    print(f" Regenerated knowledge_base.json")
                except Exception as e:
                    print(f"Error regenerating KB: {e}")

            #Clear cache
            from utils import clear_model_cache
            clear_model_cache(user_id)
            print(f" Cleared model cache for user {user_id}")

            return jsonify({
                'success': True,
                'message': 'Q&A pair deleted successfully!'
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to delete Q&A pair'})

    except Exception as e:
        print(f"Delete Q&A error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


#Download intents
@app.route('/chatbot/download-intents/<int:chatbot_id>', methods=['GET'])
def download_intents(chatbot_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    chatbot = Chatbot.query.get_or_404(chatbot_id)

    if chatbot.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    if not chatbot.training_data:
        return jsonify({'success': False, 'message': 'No training data available'}), 404

    try:
        training_data = json.loads(chatbot.training_data)

        from io import BytesIO
        json_data = BytesIO()
        json_data.write(json.dumps(training_data, indent=2, ensure_ascii=False).encode('utf-8'))
        json_data.seek(0)

        #Sanitize filename for cross-platform compatibility
        safe_name = "".join(c for c in chatbot.name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_name}_intents.json"
        print("DOWNLOAD FILE NAME:", filename)

        return send_file(
            json_data,
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Download error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Failed to download intents'}), 500


@app.route('/chatbot/get-qa-pairs/<int:chatbot_id>', methods=['GET'])
def get_qa_pairs(chatbot_id):
    """Get Q&A pairs with timestamps"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    chatbot = db.session.get(Chatbot, chatbot_id)

    if chatbot.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    try:
        from database import get_qa_pairs_from_db
        qa_pairs = get_qa_pairs_from_db(chatbot_id)

        #If no database records, try training_data JSON
        if not qa_pairs and chatbot.training_data:
            training_data = json.loads(chatbot.training_data)
            intents = training_data.get('intents', [])

            qa_pairs = []
            for intent in intents:
                patterns = intent.get('patterns', [])
                responses = intent.get('responses', [])

                if patterns and responses:
                    qa_pairs.append({
                        'id': None,
                        'question': patterns[0],
                        'answer': responses[0],
                        'tag': intent.get('tag', ''),
                        'created_at': None,
                        'updated_at': None
                    })

        return jsonify({
            'success': True,
            'qa_pairs': qa_pairs,
            'count': len(qa_pairs)
        })

    except Exception as e:
        print(f"Get Q&A error: {e}")
        return jsonify({'success': False, 'message': 'Failed to load Q&A pairs'})


# UPDATE Q&A PAIR
@app.route('/chatbot/update-qa-pair/<int:qa_id>', methods=['POST'])
def update_qa_pair_enhanced(qa_id):
    """Update Q&A pair with chatbot-specific regeneration"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    try:
        data = request.get_json()
        question = data.get('question')
        answer = data.get('answer')

        if not question and not answer:
            return jsonify({'success': False, 'message': 'Question or answer required'})

        qa_pair = QAPair.query.get(qa_id)
        if not qa_pair:
            return jsonify({'success': False, 'message': 'Q&A pair not found'})

        chatbot = Chatbot.query.get(qa_pair.chatbot_id)
        if chatbot.user_id != session['user_id']:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403

        # Update fields
        if question:
            if len(question) > 500:
                return jsonify({'success': False, 'message': 'Question exceeds 500 characters'})
            qa_pair.question = question.strip()

        if answer:
            if len(answer) > 1000:
                return jsonify({'success': False, 'message': 'Answer exceeds 1000 characters'})
            qa_pair.answer = answer.strip()

        qa_pair.updated_at = datetime.utcnow()
        db.session.commit()

        #Regenerate chatbot-specific intents
        regenerate_intents_from_db(chatbot.id)

        #Clear chatbot-specific cache
        from utils import clear_model_cache
        clear_model_cache(chatbot.user_id, chatbot_id=chatbot.id)

        return jsonify({
            'success': True,
            'message': 'Q&A pair updated successfully',
            'qa_pair': qa_pair.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        print(f" Update error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500



@app.route('/chatbot/delete-training/<int:chatbot_id>', methods=['POST'])
def delete_training_data(chatbot_id):
    """Delete training data with chatbot isolation"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401

    chatbot = Chatbot.query.get_or_404(chatbot_id)

    if chatbot.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    try:
        #Delete QA pairs from database
        QAPair.query.filter_by(chatbot_id=chatbot_id).delete()

        #Clear chatbot training data
        chatbot.training_data = None
        chatbot.training_file = None
        chatbot.is_trained = False

        db.session.commit()

        #Delete chatbot-specific files
        chatbot_folder = get_chatbot_folder(chatbot.user_id, chatbot_id)
        if os.path.exists(chatbot_folder):
            import shutil
            shutil.rmtree(chatbot_folder)
            print(f"Deleted chatbot folder: {chatbot_folder}")

        #Clear chatbot-specific cache
        from utils import clear_model_cache
        clear_model_cache(chatbot.user_id, chatbot_id=chatbot_id)

        return jsonify({
            'success': True,
            'message': 'Training data deleted successfully'
        })

    except Exception as e:
        db.session.rollback()
        print(f" Delete training error: {e}")


@app.route('/ml/train-enhanced', methods=['POST'])
def train_ml_model_enhanced():
    """Enhanced ML training with chatbot isolation"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    user_id = session['user_id']

    #Get chatbot_id from request
    data = request.get_json()
    chatbot_id = data.get('chatbot_id') if data else None

    if not chatbot_id:
        return jsonify({'error': 'Chatbot ID required'}), 400

    #Verify ownership
    chatbot = Chatbot.query.get(chatbot_id)
    if not chatbot or chatbot.user_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403

    #Check chatbot-specific intents path
    chatbot_folder = get_chatbot_folder(chatbot.user_id, chatbot_id)
    intents_path = os.path.join(chatbot_folder, 'intents.json')

    if not os.path.exists(intents_path):
        return jsonify({'error': 'Please upload training data first (intents.json)'}), 400

    try:
        from model_training import train_chatbot_model
        from utils import clear_model_cache

        print(f"\n{'=' * 60}")
        print(f"  Starting Enhanced AI Training for Chatbot {chatbot_id}")
        print(f"{'=' * 60}\n")

        #Use chatbot-specific training
        result = train_chatbot_model(user_id, chatbot_id)
        clear_model_cache(user_id, chatbot_id=chatbot_id)

        return jsonify({
            'success': True,
            'message': f' Enhanced AI model trained successfully with {result["accuracy"]:.1%} accuracy!',
            'details': result,
            'chatbot_id': chatbot_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


@app.route('/data/users/<path:filepath>')
def serve_user_data(filepath):
    """Serve user data files with proper MIME types and security"""
    try:
        # Construct full path
        full_path = os.path.join('data', 'users', filepath)

        # Security check - prevent directory traversal attacks
        abs_data_path = os.path.abspath('data/users')
        abs_requested_path = os.path.abspath(full_path)

        if not abs_requested_path.startswith(abs_data_path):
            print(f" SECURITY: Path outside data/users: {filepath}")
            return "Unauthorized", 403

        if not os.path.exists(full_path):
            print(f" File not found: {full_path}")
            return "File not found", 404

        # Determine MIME type
        import mimetypes
        mime_type, _ = mimetypes.guess_type(full_path)

        # Default MIME types for common image formats
        if not mime_type:
            ext = os.path.splitext(full_path)[1].lower()
            mime_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
                '.webp': 'image/webp'
            }
            mime_type = mime_map.get(ext, 'application/octet-stream')

        print(f" Serving: {full_path} ({mime_type})")

        return send_file(
            full_path,
            mimetype=mime_type,
            as_attachment=False,
            download_name=os.path.basename(full_path)
        )

    except Exception as e:
        print(f" Error serving file: {e}")
        import traceback
        traceback.print_exc()
        return "Internal server error", 500


@app.route('/chatbot/analytics/<int:chatbot_id>')
@subscription_required
def chatbot_analytics(chatbot_id):
        """View chatbot analytics and conversation history"""
        if 'user_id' not in session:
            return redirect(url_for('login'))

        chatbot = Chatbot.query.get_or_404(chatbot_id)

        # Check authorization
        if chatbot.user_id != session['user_id']:
            flash('Unauthorized access', 'error')
            return redirect(url_for('dashboard'))

        # Get analytics data
        from database import get_chatbot_analytics, get_chatbot_sessions

        analytics_7d = get_chatbot_analytics(chatbot_id, days=7)
        analytics_30d = get_chatbot_analytics(chatbot_id, days=30)
        recent_sessions = get_chatbot_sessions(chatbot_id, limit=20)

        user = User.query.get(session['user_id'])
        chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()

        return render_template(
            'analytics.html',
            chatbot=chatbot,
            analytics_7d=analytics_7d,
            analytics_30d=analytics_30d,
            recent_sessions=recent_sessions,
            chatbots=chatbots,
            user=user
        )

    #SESSION DETAILS
@app.route('/chatbot/session/<int:session_id>')
@subscription_required
def view_session(session_id):
        """View detailed conversation history for a session"""
        if 'user_id' not in session:
            return redirect(url_for('login'))

        chat_session = ChatSession.query.get_or_404(session_id)
        chatbot = Chatbot.query.get(chat_session.chatbot_id)

        # Check authorization
        if chatbot.user_id != session['user_id']:
            flash('Unauthorized access', 'error')
            return redirect(url_for('dashboard'))

        messages = get_session_messages(session_id)

        user = User.query.get(session['user_id'])
        chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()

        return render_template(
            'session_details.html',
            chat_session=chat_session,
            chatbot=chatbot,
            messages=messages,
            chatbots=chatbots,
            user=user
        )


    #GET MODEL INFO
@app.route('/api/model-info/<int:user_id>')
def get_model_info_api(user_id):
        """Get information about trained model"""
        if 'user_id' not in session or session['user_id'] != user_id:
            return jsonify({'error': 'Unauthorized'}), 401

        from utils import get_model_info

        info = get_model_info(user_id)
        return jsonify(info)


@app.route('/chatbot/preview/<int:chatbot_id>')
def preview_chatbot(chatbot_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    chatbot = db.session.get(Chatbot, chatbot_id)

    if not chatbot or chatbot.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))

    user = db.session.get(User, session['user_id'])
    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()

    # Parse welcome buttons
    welcome_buttons_data = []
    if chatbot.welcome_buttons:
        try:
            welcome_buttons_data = json.loads(chatbot.welcome_buttons)
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing welcome buttons: {e}")
            welcome_buttons_data = []

    # ✅ FIX: Use centralized avatar URL function
    bot_avatar_url = get_avatar_url(chatbot, request)

    return render_template(
        'preview_chatbot.html',
        chatbot=chatbot,
        chatbots=chatbots,
        user=user,
        welcome_buttons=welcome_buttons_data,
        bot_avatar_url=bot_avatar_url  # Pass validated URL
    )

#Deploy Chatbot
@app.route('/chatbot/deploy/<int:chatbot_id>')
def deploy_chatbot(chatbot_id):
    """Deploy/activate a chatbot"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    chatbot = Chatbot.query.get_or_404(chatbot_id)

    if chatbot.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))

    # Activate
    chatbot.is_active = True
    db.session.commit()

    # Generate iframe code
    embed_url = f"{request.host_url.rstrip('/')}/embed/{chatbot.embed_code}"
    iframe_code = f'<iframe src="{embed_url}" width="400" height="600" frameborder="0" style="border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></iframe>'

    user = db.session.get(User, session['user_id'])
    chatbots = Chatbot.query.filter_by(user_id=user.id).order_by(Chatbot.created_at.desc()).all()

    flash('Chatbot deployed successfully!', 'success')

    return render_template(
        'deploy_chatbot.html',
        chatbot=chatbot,
        iframe_code=iframe_code,  #Pass as plain text
        chatbots=chatbots,
        user=user
    )
#Toggle Chatbot
@app.route('/chatbot/toggle/<int:chatbot_id>')
def toggle_chatbot(chatbot_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'})

    chatbot = db.session.get(Chatbot, chatbot_id)

    if chatbot.user_id != session['user_id']:
        return jsonify({'success': False, 'message': 'Unauthorized'})

    chatbot.is_active = not chatbot.is_active
    db.session.commit()

    return jsonify({'success': True, 'is_active': chatbot.is_active})


@app.route('/chatbot/delete/<int:chatbot_id>')
def delete_chatbot(chatbot_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])
    chatbot = db.session.get(Chatbot, chatbot_id)

    if chatbot.user_id != session['user_id']:
        flash('Unauthorized access', 'error')
        return redirect(url_for('dashboard'))

    if chatbot.training_file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], chatbot.training_file)
        if os.path.exists(filepath):
            os.remove(filepath)

    db.session.delete(chatbot)

    #Use the new decrement method
    if user.subscription:
        user.subscription.decrement_chatbot_count()

    db.session.commit()
    flash('Chatbot deleted successfully!', 'success')
    return redirect(url_for('dashboard'))



#EMBED ROUTE
@app.route('/embed/<embed_code>')
def embed_chatbot(embed_code):
    """Embed route with proper avatar URL handling"""
    print(f"\n{'=' * 70}")
    print(f" EMBED ROUTE CALLED")
    print(f"   Embed Code: {embed_code}")
    print(f"{'=' * 70}\n")

    chatbot = Chatbot.query.filter_by(embed_code=embed_code).first()

    if not chatbot:
        print(f" Chatbot not found: {embed_code}")
        return render_template('embed_error.html',
                               error="Chatbot not found",
                               message="This chatbot may have been removed or the link is incorrect."), 404

    if not chatbot.is_active:
        print(f" Chatbot not active: {chatbot.name}")
        return render_template('embed_error.html',
                               error="Chatbot unavailable",
                               message="This chatbot is currently inactive."), 403

    print(f" Chatbot found: {chatbot.name}")
    print(f"   ID: {chatbot.id}")
    print(f"   User ID: {chatbot.user_id}")

    #Parse welcome buttons with full submenu support
    welcome_buttons_data = []
    if chatbot.welcome_buttons:
        try:
            welcome_buttons_data = json.loads(chatbot.welcome_buttons)
            print(f"   Welcome buttons: {len(welcome_buttons_data)}")
            for btn in welcome_buttons_data:
                print(f"   - {btn.get('text')} ({btn.get('type')})")
                if btn.get('has_submenu'):
                    submenu_items = btn.get('submenu_items', [])
                    print(f"     └─ {len(submenu_items)} submenu items")
        except json.JSONDecodeError as e:
            print(f" Error parsing welcome buttons: {e}")
            welcome_buttons_data = []

    #Proper Avatar URL Handling
    bot_avatar_url = None

    if chatbot.bot_avatar:
        avatar_path = chatbot.bot_avatar.strip()
        base_url = request.url_root.rstrip('/')

        print(f"\n{'=' * 70}")
        print(f"  AVATAR PROCESSING")
        print(f"   Raw Path: {avatar_path}")
        print(f"   Base URL: {base_url}")
        print(f"{'=' * 70}")

        # Case 1: Already a full URL (http:// or https://)
        if avatar_path.startswith(('http://', 'https://')):
            bot_avatar_url = avatar_path
            print(f"   ✓ Full URL detected")

        # Case 2: Path starts with /data/users/ (chatbot-specific folder)
        elif avatar_path.startswith('/data/users/'):
            bot_avatar_url = base_url + avatar_path
            print(f"   ✓ Chatbot folder path")

        # Case 3: Path starts with /static/ (legacy)
        elif avatar_path.startswith('/static/'):
            bot_avatar_url = base_url + avatar_path
            print(f"   ✓ Static folder path")

        # Case 4: Relative path without leading slash
        else:
            #Ensure it starts with /data/users/
            if not avatar_path.startswith('data/users/'):
                avatar_path = f"data/users/{avatar_path.lstrip('/')}"
            bot_avatar_url = f"{base_url}/{avatar_path}"
            print(f"   ✓ Relative path converted")

        print(f"   Final URL: {bot_avatar_url}")
        print(f"{'=' * 70}\n")

        #Verify file exists
        if bot_avatar_url.startswith(base_url):
            local_path = bot_avatar_url.replace(base_url + '/', '')
            if not os.path.exists(local_path):
                print(f"     Warning: Avatar file not found at {local_path}")
                bot_avatar_url = None  #Fallback to default icon
    else:
        print(f"     No avatar uploaded - will use default message icon")

    #Prepare all settings
    chatbot_settings = {
        'embedCode': chatbot.embed_code,
        'botName': chatbot.bot_name or 'AI Assistant',
        'welcomeMessage': chatbot.welcome_message or 'Hello! How can I help you?',
        'themeColor': chatbot.theme_color or '#4F46E5',
        'chatBackgroundColor': chatbot.chat_background_color or '#F7FAFC',
        'botMessageColor': chatbot.bot_message_color or '#FFFFFF',
        'botTextColor': chatbot.bot_text_color or '#1A202C',
        'userMessageColor': chatbot.user_message_color or (chatbot.theme_color or '#4F46E5'),
        'userTextColor': chatbot.user_text_color or '#FFFFFF',
        'botAvatar': bot_avatar_url,  #Properly formatted URL or None
        'welcomeButtons': welcome_buttons_data
    }

    print(f"\n{'=' * 70}")
    print(f" RENDERING EMBED:")
    print(f"   Settings: ")
    print(f"   Avatar: {' ' + bot_avatar_url[:50] if bot_avatar_url else ' (default message icon)'}")
    print(f"   Buttons: {len(welcome_buttons_data)}")
    print(f"{'=' * 70}\n")

    return render_template(
        'embed.html',
        chatbot=chatbot,
        settings=chatbot_settings,
        welcome_buttons=welcome_buttons_data,
        bot_avatar_url=bot_avatar_url  #Pass properly formatted URL
    )


@app.route('/api/chat/<embed_code>', methods=['POST'])
def chat_api(embed_code):
    """ CHAT API - Enhanced with Smart Fallback Detection"""
    start_time = time.time()

    try:
        # STEP 1: Validate chatbot
        chatbot = Chatbot.query.filter_by(embed_code=embed_code).first()

        if not chatbot:
            return jsonify({'error': 'Chatbot not found'}), 404

        user = db.session.get(User, chatbot.user_id)
        if not user or not user.subscription:
            return jsonify({'error': 'Service unavailable'}), 503

        # STEP 2: Check subscription limits
        if not user.subscription.can_send_message():
            return jsonify({
                'error': 'Message limit reached',
                'message': 'The chatbot owner has reached their monthly message limit.'
            }), 429

        # STEP 3: Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')
        button_action = data.get('button_action')
        is_voice = data.get('is_voice', False)  # NEW: Support voice mode

        print(f"\n{'=' * 70}")
        print(f" CHAT API REQUEST")
        print(f"   Chatbot: {chatbot.name} (ID: {chatbot.id})")
        print(f"   Message: {user_message[:50] if user_message else 'N/A'}...")
        print(f"   Session: {session_id}")
        print(f"   Button Action: {button_action is not None}")
        print(f"   Voice Mode: {is_voice}")
        print(f"{'=' * 70}\n")

        if not user_message and not button_action:
            return jsonify({'error': 'Empty message'}), 400

        # STEP 4: Session management
        if not session_id:
            user_ip = request.remote_addr
            user_agent = request.headers.get('User-Agent', 'Unknown')
            session_id, session_token = create_session(
                chatbot_id=chatbot.id,
                ip=user_ip,
                user_agent=user_agent
            )
        else:
            session_token = None

        # STEP 5: Handle button action
        if button_action:
            action_type = button_action.get('type')
            action_value = button_action.get('value')

            bot_response = process_button_action(action_type, action_value, chatbot)
            user_message = f"[Button: {button_action.get('text', action_value)}]"
            intent = f'button_{action_type}'
            confidence = 0.95
            is_fallback = False
        else:
            # ✅ LAZY IMPORT utils only when chat happens
            from utils import get_smart_response, get_intent_with_confidence

            # STEP 6: Get AI response with chatbot_id and voice mode
            bot_response = get_smart_response(
                user_message,
                chatbot.user_id,
                chatbot_id=chatbot.id,
                session_id=str(session_id),
                is_voice=is_voice  # NEW: Pass voice mode
            )

            # Get intent and confidence
            intent, confidence = get_intent_with_confidence(
                user_message,
                chatbot.user_id,
                chatbot_id=chatbot.id
            )

            # ENHANCED: Detect all smart fallback responses
            is_fallback = detect_fallback_response(bot_response, confidence)

        # STEP 7: Log conversation
        processing_time = int((time.time() - start_time) * 1000)

        try:
            log_message(
                session_id=session_id,
                sender='user',
                message=user_message,
                intent=intent or 'unknown',
                confidence=confidence
            )

            log_message(
                session_id=session_id,
                sender='bot',
                message=bot_response,
                intent=intent or 'unknown',
                confidence=confidence,
                is_fallback=is_fallback,
                processing_time=processing_time
            )

            # Increment message counter
            if user.subscription:
                user.subscription.increment_message_count()

        except Exception as e:
            print(f" Logging error: {e}")

        # STEP 8: Return response
        response_data = {
            'response': bot_response,
            'session_id': session_id,
            'intent': intent or 'unknown',
            'confidence': round(confidence, 2),
            'processing_time_ms': processing_time,
            'is_fallback': is_fallback,
            'chatbot_id': chatbot.id,
            'chatbot_name': chatbot.bot_name
        }

        if session_token:
            response_data['session_token'] = session_token

        print(f" Response sent ({processing_time}ms)\n")
        return jsonify(response_data)

    except Exception as e:
        print(f" CHAT API ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'error': 'Internal server error',
            'message': 'Sorry, something went wrong. Please try again.',
            'details': str(e) if app.debug else None
        }), 500


# Add at the very end, BEFORE if __name__ == '__main__':
# application = app  # WSGI servers look for 'application'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if SubscriptionPlan.query.count() == 0:
            initialize_subscription_plans()

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    app.run(host=host, port=port, debug=debug)