"""
Gunicorn configuration for production deployment
Optimized for ML-heavy Flask application with proper timeout handling
"""

import os
import multiprocessing

# SERVER SOCKET CONFIGURATION
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5000')}"
backlog = 2048

# WORKER CONFIGURATION
# Calculate workers based on CPU cores
cpu_count = multiprocessing.cpu_count()
workers = int(os.getenv('WORKERS', min(cpu_count * 2 + 1, 8)))  # Cap at 8 workers

# Worker class - use sync for ML-heavy operations
worker_class = 'sync'
worker_connections = 1000

# TIMEOUT CONFIGURATION
timeout = 180              # 3 minutes - worker timeout
graceful_timeout = 120     # 2 minutes - graceful shutdown
keepalive = 5              # Keep-alive connections

# RESTART & RESOURCE MANAGEMENT
max_requests = 1000           # Restart worker after 1000 requests
max_requests_jitter = 50      # Add randomness to avoid all workers restarting at once

# Load app before forking workers - shares memory for ML models
preload_app = True

# LOGGING CONFIGURATION
log_dir = os.getenv('LOG_DIR', '/var/www/chatbot_panel/logs')
os.makedirs(log_dir, exist_ok=True)

accesslog = os.path.join(log_dir, 'access.log')
errorlog = os.path.join(log_dir, 'error.log')
loglevel = os.getenv('LOG_LEVEL', 'info').lower()

# Detailed access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# PROCESS NAMING
proc_name = 'chatbot_panel'

# SERVER MECHANICS
daemon = False
pidfile = '/tmp/chatbot_panel.pid'
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL CONFIGURATION (Optional)
if os.getenv('SSL_ENABLED', 'False').lower() == 'true':
    certfile = os.getenv('SSL_CERT_PATH')
    keyfile = os.getenv('SSL_KEY_PATH')

# WORKER LIFECYCLE HOOKS (For Monitoring)
def on_starting(server):
    """Called just before the master process is initialized"""
    print(f" Starting Gunicorn with {workers} workers")
    print(f"  Timeout set to {timeout}s for ML model loading")

def on_reload(server):
    """Called when configuration is reloaded"""
    print(" Reloading configuration")

def when_ready(server):
    """Called just after the server is started"""
    print(f" Gunicorn ready on {bind}")
    print(f"Workers: {workers} | Timeout: {timeout}s | Preload: {preload_app}")

def worker_int(worker):
    """Called when worker receives INT or QUIT signal"""
    print(f"  Worker {worker.pid} interrupted")

def worker_abort(worker):
    """Called when worker times out"""
    print(f" Worker {worker.pid} aborted (timeout)")

# DEVELOPMENT vs PRODUCTION SETTINGS
if os.getenv('FLASK_ENV') == 'development':
    reload = True           # Auto-reload on code changes
    workers = 2             # Fewer workers for dev
    loglevel = 'debug'
else:
    reload = False