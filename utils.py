import json
import os
import pickle
import random
from pathlib import Path
from dotenv import load_dotenv
import string
import re
import hashlib
import time
import uuid
from html import unescape
from typing import Dict, Optional, Any, List, Tuple
from threading import RLock
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps
import hashlib
from functools import lru_cache

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Response cache - stores recent responses
RESPONSE_CACHE = {}
CACHE_MAX_SIZE = 5000
CACHE_TTL = 1800  # 30 minutes


def get_cached_response(cache_key):
    """Get cached response if available"""
    if cache_key in RESPONSE_CACHE:
        cached_data, timestamp = RESPONSE_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
        else:
            del RESPONSE_CACHE[cache_key]
    return None


def set_cached_response(cache_key, response):
    """Cache a response"""
    # Clean old cache if too large
    if len(RESPONSE_CACHE) > CACHE_MAX_SIZE:
        # Remove 20% oldest entries
        sorted_keys = sorted(RESPONSE_CACHE.items(), key=lambda x: x[1][1])
        for key, _ in sorted_keys[:int(CACHE_MAX_SIZE * 0.2)]:
            del RESPONSE_CACHE[key]

    RESPONSE_CACHE[cache_key] = (response, time.time())


# FUZZY MATCHING IMPORTS
try:
    from difflib import SequenceMatcher
    from Levenshtein import distance as levenshtein_distance

    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    print(" python-Levenshtein not available - using basic fuzzy matching")

# Existing imports
SENTENCE_TRANSFORMERS_AVAILABLE = False
OLLAMA_AVAILABLE = False
SKLEARN_AVAILABLE = False

# ✅ LAZY IMPORT: Check availability but don't load heavy models yet
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✓ Sentence Transformers available (lazy load)")
except ImportError:
    print("⚠ Sentence Transformers not available")

SKLEARN_AVAILABLE = False
try:
    import sklearn
    SKLEARN_AVAILABLE = True
    print("✓ Sklearn available (lazy load)")
except ImportError:
    print("⚠ Sklearn not available")

def _lazy_import_sklearn():
    """Lazy import sklearn when needed"""
    global cosine_similarity, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import ollama

    # Set Ollama host from environment
    ollama_host = os.getenv('OLLAMA_HOST')
    if ollama_host:
        os.environ['OLLAMA_HOST'] = ollama_host

    # Just test connection instead of checking models
    ollama.list()
    OLLAMA_AVAILABLE = True
    print("✓ Ollama available")

except Exception as e:
    OLLAMA_AVAILABLE = False
    print(" Ollama not available:", e)


# Configuration from environment
MAX_WORKERS = int(os.getenv('WORKERS', '50'))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '25'))
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '8'))
CACHE_TTL = int(os.getenv('ML_CACHE_TTL', '1800'))
MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '5000'))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '60'))
MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '120'))

# Ollama models from environment
ollama_models_str = os.getenv('OLLAMA_MODELS', 'llama2,mistral,neural-chat,orca-mini')
OLLAMA_MODELS = [model.strip() for model in ollama_models_str.split(',')]

# Override OLLAMA_AVAILABLE from environment if explicitly set
if os.getenv('OLLAMA_AVAILABLE', '').lower() == 'false':
    OLLAMA_AVAILABLE = False

CONFIDENCE_THRESHOLDS = {
    'exact_match': 1.0,  # Perfect string match
    'fuzzy_match': 0.90,  # Typo-tolerant
    'semantic_perfect': 0.85,  # Very high semantic similarity
    'hybrid_confirmed': 0.75,  # Embeddings + SVM agree
    'high': 0.60,  # High confidence - strong KB match
    'medium': 0.40,  # Medium confidence - still use KB if available
    'low': 0.25,  # Low confidence - probably not in KB
    'fallback': 0.0
}


# FUZZY MATCHING UTILITIES
def calculate_levenshtein_distance(str1: str, str2: str) -> int:
    """
    Calculate edit distance between two strings
    Uses optimized library if available, else fallback
    """
    if LEVENSHTEIN_AVAILABLE:
        return levenshtein_distance(str1, str2)
    else:
        # Fallback implementation
        if len(str1) < len(str2):
            return calculate_levenshtein_distance(str2, str1)

        if len(str2) == 0:
            return len(str1)

        previous_row = range(len(str2) + 1)
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


def character_ngram_similarity(str1: str, str2: str, n: int = 2) -> float:
    """
    Calculate similarity using character n-grams
    Useful for partial word matches
    """

    def get_ngrams(text: str, n: int) -> set:
        text = text.lower().strip()
        return set(text[i:i + n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(str1, n)
    ngrams2 = get_ngrams(str2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(intersection) / len(union) if union else 0.0


def phonetic_similarity(word1: str, word2: str) -> bool:
    """
    Simple phonetic matching for sound-alike words
    Returns True if words sound similar
    """

    def soundex(word: str) -> str:
        """Simple Soundex algorithm"""
        word = word.upper()
        soundex_map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }

        if not word:
            return ""

        soundex_code = word[0]
        for char in word[1:]:
            code = soundex_map.get(char, '0')
            if code != '0' and code != soundex_code[-1]:
                soundex_code += code

        soundex_code += '000'
        return soundex_code[:4]

    return soundex(word1) == soundex(word2)


def enhanced_pattern_match(user_text: str, pattern: str,
                           allow_typos: bool = True,
                           max_typos: int = 2) -> Tuple[float, str]:
    """
    Enhanced pattern matching with typo tolerance

    Returns:
        (score: float, match_type: str)
        match_type: 'exact', 'fuzzy', 'partial', 'ngram', 'none'
    """
    user_clean = user_text.lower().strip()
    pattern_clean = pattern.lower().strip()

    # 1. Exact match
    if user_clean == pattern_clean:
        return 1.0, 'exact'

    # 2. Fuzzy match (typo tolerance)
    if allow_typos:
        fuzzy_score = fuzzy_match_score(user_clean, pattern_clean, max_typos)
        if fuzzy_score >= 0.85:
            return fuzzy_score, 'fuzzy'

    # 3. Word-level matching
    user_words = set(user_clean.split())
    pattern_words = set(pattern_clean.split())

    if not user_words or not pattern_words:
        return 0.0, 'none'

    # Calculate Jaccard similarity
    intersection = user_words & pattern_words
    union = user_words | pattern_words
    jaccard = len(intersection) / len(union) if union else 0.0

    # 4. Check for fuzzy word matches
    if allow_typos and jaccard < 0.5:
        fuzzy_word_matches = 0
        for u_word in user_words:
            for p_word in pattern_words:
                # Allow 1-character difference for words > 4 chars
                if len(u_word) > 4 and len(p_word) > 4:
                    if calculate_levenshtein_distance(u_word, p_word) <= 1:
                        fuzzy_word_matches += 1
                        break

        if fuzzy_word_matches > 0:
            jaccard += (fuzzy_word_matches / len(pattern_words)) * 0.3

    # 5. Boost for substring matches
    if pattern_clean in user_clean:
        jaccard += 0.25
    elif user_clean in pattern_clean:
        jaccard += 0.20

    # 6. N-gram similarity for partial matches
    ngram_score = character_ngram_similarity(user_clean, pattern_clean, n=3)
    if ngram_score > 0.5:
        jaccard = max(jaccard, ngram_score * 0.8)

    # 7. Boost for high word overlap
    overlap_ratio = len(intersection) / len(pattern_words) if pattern_words else 0
    if overlap_ratio > 0.7:
        jaccard += 0.15

    final_score = min(1.0, jaccard)

    if final_score >= 0.7:
        match_type = 'partial'
    elif final_score >= 0.4:
        match_type = 'ngram'
    else:
        match_type = 'none'

    return final_score, match_type


def _safe_cache_set(user_id: int, key: str, value: tuple, chatbot_id: int, ttl: int = 1800):
    """Safely set cache with error handling"""
    try:
        SMART_CACHE.set(user_id, key, value, session_id=str(chatbot_id), ttl=ttl)
    except Exception as e:
        logger.error(f"Cache set error: {e}")


# ============================================================================
# ML MODEL CACHE (Unchanged from original)
# ============================================================================
class MLModelCache:
    """Cache trained ML models in memory"""

    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.lock = RLock()

    def get_model(self, user_id: int, chatbot_id: int, model_type: str = 'svm'):
        """✅ OPTIMIZED: Load and cache ML model with faster access"""
        cache_key = f"{user_id}_{chatbot_id}_{model_type}"

        with self.lock:
            # ✅ OPTIMIZED: Check cache first (fast path)
            if cache_key in self.models:
                cached = self.models[cache_key]
                if time.time() - cached['loaded_at'] < 3600:  # 1 hour cache
                    return cached['model'], cached['vectorizer'], cached['label_encoder']

            # Load from disk
            base_folder = os.getenv('BASE_DATA_FOLDER', 'data')
            models_dir = os.path.join(base_folder, 'users', f'user_{user_id}',
                                      'chatbots', f'chatbot_{chatbot_id}', 'models')

            model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
            vectorizer_path = os.path.join(models_dir, f'{model_type}_vectorizer.pkl')

            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                return None, None, None

            try:
                # ✅ OPTIMIZED: Load both files in parallel would be ideal, but pickle is sequential
                # Using faster pickle protocol if available
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)

                if isinstance(model_data, dict):
                    model = model_data.get('model')
                    label_encoder = model_data.get('label_encoder')
                    # ✅ Extract training info if available for debugging
                    training_info = model_data.get('training_info', {})
                else:
                    model = model_data
                    label_encoder = None
                    training_info = {}

                # ✅ OPTIMIZED: Cache with metadata
                self.models[cache_key] = {
                    'model': model,
                    'vectorizer': vectorizer,
                    'label_encoder': label_encoder,
                    'loaded_at': time.time(),
                    'training_info': training_info
                }

                print(f"✓ Loaded {model_type} model for chatbot {chatbot_id}" +
                      (f" (acc: {training_info.get('accuracy', 0):.1%})" if training_info.get('accuracy') else ""))
                return model, vectorizer, label_encoder

            except Exception as e:
                print(f"⚠ Error loading {model_type} model: {e}")
                return None, None, None

    def clear_cache(self, user_id: int = None, chatbot_id: int = None):
        """Clear model cache"""
        with self.lock:
            if user_id and chatbot_id:
                to_remove = [k for k in self.models.keys()
                             if k.startswith(f"{user_id}_{chatbot_id}_")]
                for key in to_remove:
                    del self.models[key]
            elif user_id:
                to_remove = [k for k in self.models.keys()
                             if k.startswith(f"{user_id}_")]
                for key in to_remove:
                    del self.models[key]
            else:
                self.models.clear()


ML_MODEL_CACHE = MLModelCache()


# ============================================================================
# FALLBACK MANAGER (Unchanged)
# ============================================================================
class FallbackMessageManager:
    """Smart contextual fallback messages"""

    UNDERSTANDING = [
        "I'm sorry, I didn't fully understand that. Could you please rephrase?",
        "I may have missed that. Can you share more detail?",
        "Could you clarify what you're looking for?",
        "I'm not completely sure I understood. Could you explain differently?"
    ]

    OUT_OF_SCOPE = [
        "That's a great question, but I don't have that information right now.",
        "I'm unable to help with that request at the moment.",
        "This topic is outside my current capabilities.",
        "I don't have access to that information yet."
    ]

    RETRY = [
        "Try asking in another way.",
        "Please use simpler or more specific keywords.",
        "Could you break your request into smaller parts?",
        "Let me know what outcome you're expecting, and I'll try again."
    ]

    ERROR = [
        "Apologies, I couldn't process that request.",
        "Something went wrong. Let's try again.",
        "I had trouble with that request.",
        "Please give it another shot."
    ]

    ESCALATION = [
        "Would you like to connect with a support representative?",
        "For further help, please contact our support team.",
        "I can help you reach a human agent if needed.",
        "I recommend reaching out to our support team for better assistance."
    ]

    VOICE = [
        "Sorry, I didn't catch that.",
        "Can you please repeat?",
        "I'm not sure I understood. Please try again.",
        "One more time, please?"
    ]

    LOW_CONFIDENCE = [
        "I'm not entirely confident. Could you clarify?",
        "I need more information for an accurate answer.",
        "Could you elaborate a bit more?",
        "To help you correctly, please provide more context."
    ]

    NO_MATCH = [
        "I understand your question, but don't have specific info on that.",
        "That's clear—but it's not in my current knowledge base.",
        "I see what you're asking, but that topic isn't covered yet.",
        "Good question! But I don't have details about that right now."
    ]

    def __init__(self):
        self.user_failure_count = defaultdict(int)
        self.lock = RLock()

    def get_fallback(self, situation: str, user_id: int = None,
                     session_id: str = None, confidence: float = 0.0,
                     is_voice: bool = False) -> str:
        """Get contextual fallback message"""

        failure_key = f"{user_id}_{session_id}" if user_id and session_id else str(user_id or "anon")

        with self.lock:
            self.user_failure_count[failure_key] += 1
            failure_count = self.user_failure_count[failure_key]

        if failure_count >= 3:
            with self.lock:
                self.user_failure_count[failure_key] = 0
            return random.choice(self.ESCALATION)

        if is_voice:
            return random.choice(self.VOICE)

        messages = {
            'understanding': self.UNDERSTANDING,
            'out_of_scope': self.OUT_OF_SCOPE,
            'retry': self.RETRY,
            'error': self.ERROR,
            'escalation': self.ESCALATION,
            'voice': self.VOICE,
            'low_confidence': self.LOW_CONFIDENCE,
            'no_match': self.NO_MATCH
        }.get(situation, self.UNDERSTANDING)

        return random.choice(messages)

    def get_smart_fallback(self, confidence: float, kb_available: bool,
                           ollama_failed: bool, user_id: int = None,
                           session_id: str = None, is_voice: bool = False) -> str:
        """Smart fallback based on context"""

        failure_key = f"{user_id}_{session_id}" if user_id and session_id else str(user_id or "anon")
        with self.lock:
            failure_count = self.user_failure_count.get(failure_key, 0)

        if failure_count >= 3:
            return self.get_fallback('escalation', user_id, session_id, confidence, is_voice)

        if is_voice:
            return self.get_fallback('voice', user_id, session_id, confidence, is_voice)

        if confidence < CONFIDENCE_THRESHOLDS['low']:
            return self.get_fallback('understanding', user_id, session_id, confidence, is_voice)

        if confidence < CONFIDENCE_THRESHOLDS['medium']:
            return self.get_fallback('low_confidence', user_id, session_id, confidence, is_voice)

        if confidence >= CONFIDENCE_THRESHOLDS['medium'] and not kb_available:
            return self.get_fallback('no_match', user_id, session_id, confidence, is_voice)

        if ollama_failed and not kb_available:
            return self.get_fallback('out_of_scope', user_id, session_id, confidence, is_voice)

        return self.get_fallback('retry', user_id, session_id, confidence, is_voice)

    def reset_failure_count(self, user_id: int, session_id: str = None):
        """Reset when successful response given"""
        failure_key = f"{user_id}_{session_id}" if session_id else str(user_id)
        with self.lock:
            self.user_failure_count[failure_key] = 0


FALLBACK_MANAGER = FallbackMessageManager()

# THREAD POOL
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="chatbot_worker")


def with_timeout(timeout: int = 10):
    """Enhanced timeout decorator"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FutureTimeoutError:
                print(f"⏱ {func.__name__} timeout after {timeout}s")
                return None
            except Exception as e:
                print(f"❌ {func.__name__} error: {e}")
                return None

        return wrapper

    return decorator


# REQUEST TRACKER
class RequestTracker:
    def __init__(self):
        self.active_requests: Dict[str, Dict] = {}
        self.user_requests: Dict[int, List[str]] = defaultdict(list)
        self.lock = RLock()

    def start_request(self, user_id: int, session_id: str, message: str) -> str:
        request_id = f"req_{user_id}_{session_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.active_requests[request_id] = {
                'user_id': user_id,
                'session_id': session_id,
                'message': message[:100],
                'start_time': time.time(),
                'status': 'processing'
            }
            self.user_requests[user_id].append(request_id)
        return request_id

    def end_request(self, request_id: str, success: bool = True):
        with self.lock:
            if request_id in self.active_requests:
                req = self.active_requests[request_id]
                req['status'] = 'completed' if success else 'failed'
                req['end_time'] = time.time()
                req['duration'] = req['end_time'] - req['start_time']
                user_id = req['user_id']
                if request_id in self.user_requests[user_id]:
                    self.user_requests[user_id].remove(request_id)
                if time.time() - req['start_time'] > 60:
                    del self.active_requests[request_id]


REQUEST_TRACKER = RequestTracker()


# SESSION MANAGER
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.lock = RLock()

    def _get_key(self, session_id: str, user_id: int) -> str:
        return f"u{user_id}_s{session_id}"

    def get_session(self, session_id: str, user_id: int) -> Optional[Dict]:
        key = self._get_key(session_id, user_id)
        with self.lock:
            return self.sessions.get(key)

    def set_session(self, session_id: str, user_id: int, data: Dict):
        key = self._get_key(session_id, user_id)
        with self.lock:
            self.sessions[key] = data
            if len(self.sessions) > MAX_CACHE_SIZE:
                self._cleanup_old()

    def _cleanup_old(self):
        with self.lock:
            items = list(self.sessions.items())
            items.sort(key=lambda x: x[1].get('created_at', 0))
            to_remove = len(items) // 10
            for key, _ in items[:to_remove]:
                del self.sessions[key]

    def add_message(self, session_id: str, user_id: int, user_msg: str, bot_msg: str):
        with self.lock:
            session = self.get_session(session_id, user_id)
            if not session:
                session = {
                    'history': [],
                    'user_id': user_id,
                    'session_id': session_id,
                    'created_at': time.time()
                }
            session['history'].append({
                'user': user_msg,
                'bot': bot_msg,
                'timestamp': time.time()
            })
            session['history'] = session['history'][-10:]
            self.set_session(session_id, user_id, session)

    def get_context(self, session_id: str, user_id: int, last_n: int = 2) -> str:
        session = self.get_session(session_id, user_id)
        if not session or 'history' not in session:
            return ""
        history = session['history'][-last_n:]
        context_parts = []
        for entry in history:
            if entry.get('user'):
                context_parts.append(f"User: {entry['user']}")
            if entry.get('bot'):
                context_parts.append(f"Assistant: {entry['bot']}")
        return "\n".join(context_parts)

    def clear_user_sessions(self, user_id: int):
        with self.lock:
            to_remove = [k for k in self.sessions.keys() if k.startswith(f"u{user_id}_")]
            for key in to_remove:
                del self.sessions[key]


SESSION_MANAGER = SessionManager()


# RATE LIMITER
class RateLimiter:
    def __init__(self):
        max_rpm = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '120'))
        self.requests: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_rpm))
        self.lock = RLock()
        self.enabled = os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true'

    def is_allowed(self, user_id: int) -> bool:
        # Skip rate limiting if disabled
        if not self.enabled:
            return True
        with self.lock:
            now = time.time()
            user_requests = self.requests[user_id]
            while user_requests and now - user_requests[0] > RATE_LIMIT_WINDOW:
                user_requests.popleft()
            if len(user_requests) >= MAX_REQUESTS_PER_MINUTE:
                return False
            user_requests.append(now)
            return True


RATE_LIMITER = RateLimiter()


# SMART CACHE
class SmartCache:
    def __init__(self):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = RLock()

    def _get_key(self, user_id: int, query: str, session_id: str = None) -> str:
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        session_part = f"_s{session_id}" if session_id else ""
        return f"u{user_id}_{query_hash}{session_part}"

    def get(self, user_id: int, query: str, session_id: str = None) -> Optional[Any]:
        key = self._get_key(user_id, query, session_id)
        with self.lock:
            if key in self.cache:
                value, expire_time = self.cache[key]
                if time.time() < expire_time:
                    return value
                else:
                    del self.cache[key]
        return None

    def set(self, user_id: int, query: str, value: Any, session_id: str = None, ttl: int = None):
        if ttl is None:
            ttl = int(os.getenv('ML_CACHE_TTL', '1800'))
        key = self._get_key(user_id, query, session_id)
        expire_time = time.time() + ttl
        with self.lock:
            self.cache[key] = (value, expire_time)
            if len(self.cache) > MAX_CACHE_SIZE:
                self._cleanup_expired()

    def _cleanup_expired(self):
        now = time.time()
        with self.lock:
            expired = [k for k, (_, exp) in self.cache.items() if exp < now]
            for k in expired:
                del self.cache[k]
            if len(self.cache) > MAX_CACHE_SIZE:
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                to_remove = len(self.cache) - (MAX_CACHE_SIZE // 2)
                for k, _ in sorted_items[:to_remove]:
                    del self.cache[k]

    def clear_user(self, user_id: int):
        with self.lock:
            to_remove = [k for k in self.cache.keys() if k.startswith(f"u{user_id}_")]
            for k in to_remove:
                del self.cache[k]


SMART_CACHE = SmartCache()


# TEXT PROCESSING
def preprocess_text(input_text: str) -> str:
    if not input_text:
        return ""
    processed = input_text.lower()
    processed = processed.translate(str.maketrans('', '', string.punctuation))
    processed = ' '.join(processed.split())
    return processed


def clean_html_tags(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


# SENTENCE TRANSFORMER
_sentence_transformer = None
_st_lock = RLock()


def get_sentence_transformer():
    """✅ LAZY LOAD: Only load sentence transformer when first needed"""
    global _sentence_transformer
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    with _st_lock:
        if _sentence_transformer is None:
            try:
                # Import only when needed
                from sentence_transformers import SentenceTransformer
                _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ Loaded sentence transformer (lazy)")
            except Exception as e:
                print(f"❌ Failed to load transformer: {e}")
                return None
        return _sentence_transformer


# KNOWLEDGE BASE OPERATIONS

@with_timeout(5)
def load_knowledge_base(user_id: int, chatbot_id: int = None) -> List[Dict]:
    """✅ Load KB with validation"""

    if not chatbot_id:
        print(f"⚠ ERROR: chatbot_id is required")
        return []

    cache_key = f"kb_{user_id}_{chatbot_id}"
    cached_kb = SMART_CACHE.get(user_id, cache_key, session_id=str(chatbot_id))
    if cached_kb:
        return cached_kb

    base_folder = os.getenv('BASE_DATA_FOLDER', 'data')
    kb_path = os.path.join(base_folder, 'users', f'user_{user_id}', 'chatbots',
                           f'chatbot_{chatbot_id}', 'knowledge_base.json')

    if not os.path.exists(kb_path):
        print(f"⚠ KB not found: {kb_path}")
        return []

    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)

        if not isinstance(kb, list):
            print(f"⚠ Invalid KB format")
            return []

        valid_kb = []
        for item in kb:
            if not isinstance(item, dict):
                continue

            intent = item.get('intent') or item.get('tag')
            patterns = item.get('patterns', [])
            responses = item.get('responses', [])

            if intent and patterns and responses:
                valid_kb.append({
                    'intent': intent,
                    'tag': intent,
                    'patterns': patterns,
                    'responses': responses,
                    'chatbot_id': chatbot_id
                })

        if not valid_kb:
            print(f"⚠ No valid intents in KB")
            return []

        SMART_CACHE.set(user_id, cache_key, valid_kb, session_id=str(chatbot_id), ttl=3600)
        print(f"✓ Loaded KB: {len(valid_kb)} intents for chatbot {chatbot_id}")
        return valid_kb

    except Exception as e:
        print(f"⚠ KB load error: {e}")
        return []


def clear_model_cache(user_id: int, chatbot_id: int = None) -> None:
    """✅ Clear all caches including ML models"""
    try:
        SMART_CACHE.clear_user(user_id)
        SESSION_MANAGER.clear_user_sessions(user_id)
        ML_MODEL_CACHE.clear_cache(user_id, chatbot_id)
        print(f"✓ Cleared all caches for user {user_id}" +
              (f", chatbot {chatbot_id}" if chatbot_id else ""))
    except Exception as e:
        print(f"⚠ Cache clear error: {e}")


@with_timeout(3)
def get_semantic_similarity(query: str, text: str, model=None) -> float:
    if not SENTENCE_TRANSFORMERS_AVAILABLE or model is None:
        return 0.0
    try:
        # ✅ LAZY IMPORT cosine_similarity
        if SKLEARN_AVAILABLE:
            _lazy_import_sklearn()

        query_emb = model.encode(query, convert_to_numpy=True)
        text_emb = model.encode(text, convert_to_numpy=True)
        similarity = cosine_similarity([query_emb], [text_emb])[0][0]
        return float(similarity)
    except Exception:
        return 0.0


# ✅ OPTIMIZED: Pattern matching cache with better performance
@lru_cache(maxsize=20000)  # ✅ Increased cache size for better hit rate
def cached_pattern_similarity(user_text: str, pattern: str) -> float:
    """✅ OPTIMIZED: Cached pattern similarity calculation with early exits"""
    user_processed = preprocess_text(user_text)
    pattern_processed = preprocess_text(pattern)

    # ✅ OPTIMIZED: Quick length check first
    len_diff = abs(len(user_processed) - len(pattern_processed))
    if len_diff > max(len(user_processed), len(pattern_processed)) * 0.6:
        return 0.0  # Too different in length

    # Exact match (fastest check)
    if user_processed == pattern_processed:
        return 1.0

    # ✅ OPTIMIZED: Quick word overlap check before expensive fuzzy
    user_words = set(user_processed.split())
    pattern_words = set(pattern_processed.split())

    if not user_words or not pattern_words:
        return 0.0

    intersection = user_words.intersection(pattern_words)
    union = user_words.union(pattern_words)
    jaccard = len(intersection) / len(union) if union else 0.0

    # ✅ OPTIMIZED: Early exit if very low similarity
    if jaccard < 0.2:
        return jaccard

        # Fuzzy match (only if word overlap is reasonable)
    if jaccard > 0.3:
        fuzzy_score = fuzzy_match_score(user_processed, pattern_processed, max_errors=2)
        if fuzzy_score >= 0.85:
            return fuzzy_score

        # Boost for substring matches
    if pattern_processed in user_processed:
        jaccard += 0.25
    elif user_processed in pattern_processed:
        jaccard += 0.20

    # Boost for high word overlap
    overlap_ratio = len(intersection) / len(pattern_words) if pattern_words else 0
    if overlap_ratio > 0.8:
        jaccard += 0.15

    return min(1.0, jaccard)


def calculate_pattern_similarity(user_text: str, pattern: str) -> float:
    """✅ Now uses cached version"""
    return cached_pattern_similarity(user_text, pattern)


def fuzzy_match_score(query: str, pattern: str, max_errors: int = 2) -> float:
    """
    Calculate fuzzy match score with typo tolerance

    Args:
        query: User input
        pattern: Pattern to match
        max_errors: Max character errors allowed (default: 2)

    Returns:
        Score 0.0-1.0 (1.0 = exact match)
    """
    query_clean = query.lower().strip()
    pattern_clean = pattern.lower().strip()

    # Exact match
    if query_clean == pattern_clean:
        return 1.0

    # Calculate edit distance
    if LEVENSHTEIN_AVAILABLE:
        from Levenshtein import distance
        edit_distance = distance(query_clean, pattern_clean)
    else:
        # Fallback: simple character difference count
        edit_distance = sum(1 for a, b in zip(query_clean, pattern_clean) if a != b)
        edit_distance += abs(len(query_clean) - len(pattern_clean))

    # Within error tolerance?
    if edit_distance <= max_errors:
        score = 1.0 - (edit_distance * 0.05)
        return max(0.85, score)  # Min 0.85 for fuzzy matches

    # Calculate similarity
    max_len = max(len(query_clean), len(pattern_clean))
    if max_len == 0:
        return 0.0

    similarity = 1.0 - (edit_distance / max_len)
    return max(0.0, similarity)


def get_ml_intent_prediction(query: str, user_id: int, chatbot_id: int) -> Tuple[Optional[str], float]:
    """✅ OPTIMIZED: Use trained ML model for intent prediction with faster inference"""

    if not SKLEARN_AVAILABLE:
        return None, 0.0

    try:
        # ✅ Try SVM first (best balance of accuracy and speed)
        model, vectorizer, label_encoder = ML_MODEL_CACHE.get_model(
            user_id, chatbot_id, 'svm'
        )

        if model and vectorizer and label_encoder:
            # ✅ OPTIMIZED: Use sparse matrix transform (faster)
            query_vector = vectorizer.transform([query])

            # ✅ OPTIMIZED: Check for calibrated model (has predict_proba)
            if hasattr(model, 'predict_proba'):
                # Calibrated model - direct probability prediction
                probabilities = model.predict_proba(query_vector)[0]
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
            elif hasattr(model, 'decision_function'):
                # LinearSVC - use decision function with sigmoid approximation
                decision_scores = model.decision_function(query_vector)[0]
                predicted_class = np.argmax(decision_scores)
                # ✅ OPTIMIZED: Better confidence from decision scores
                # Normalize decision scores to [0, 1] using softmax approximation
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probabilities = exp_scores / np.sum(exp_scores)
                confidence = float(probabilities[predicted_class])
            else:
                # Fallback: direct prediction
                predicted_class = model.predict(query_vector)[0]
                confidence = 0.75  # Default confidence

            intent = label_encoder.inverse_transform([predicted_class])[0]
            print(f"  🤖 ML (SVM): '{intent}' ({confidence:.2%})")
            return intent, float(confidence)

        # Fallback to Neural Network
        model, vectorizer, label_encoder = ML_MODEL_CACHE.get_model(
            user_id, chatbot_id, 'neural_network'
        )

        if model and vectorizer and label_encoder:
            query_vector = vectorizer.transform([query])

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(query_vector.toarray())[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
            else:
                predicted_class = model.predict(query_vector.toarray())[0]
                confidence = 0.75

            intent = label_encoder.inverse_transform([predicted_class])[0]
            print(f"  🤖 ML (NN): '{intent}' ({confidence:.2%})")
            return intent, float(confidence)

        return None, 0.0

    except Exception as e:
        print(f"⚠ ML prediction error: {e}")
        return None, 0.0


# ============================================================================
# ✅ COMPLETELY FIXED: INTENT DETECTION WITH PROPER PRIORITY
# ============================================================================
import logging
from typing import Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)


def get_intent_with_confidence(query: str, user_id: int, chatbot_id: int = None) -> Tuple[Optional[str], float]:
    """
    ✅ FIXED: Proper intent detection with correct Ollama fallback integration

    PRIORITY ORDER:
    1. Cache lookup (fastest)
    2. General conversation intents (greetings, thanks, etc.)
    3. Exact/Fuzzy match
    4. High semantic similarity (>0.65)
    5. ML model prediction (>0.45)
    6. Pattern matching
    7. Best match if confidence >= 0.30
    8. Return low confidence for Ollama fallback in get_smart_response()

    Note: This function returns (intent, confidence).
    The get_smart_response() function handles Ollama fallback when confidence is low.
    """

    if not chatbot_id:
        logger.error(f"chatbot_id required for user {user_id}")
        return None, 0.0

    if not query or not isinstance(query, str):
        logger.error(f"Invalid query: {query}")
        return None, 0.0

    query = query.strip()
    if len(query) == 0:
        return None, 0.0

    # === CACHE CHECK ===
    cache_key = f"intent:{query}_{chatbot_id}"
    try:
        cached = SMART_CACHE.get(user_id, cache_key, session_id=str(chatbot_id))
        if cached and isinstance(cached, tuple) and len(cached) == 2:
            logger.info(f"Cache hit: {cached[0]} ({cached[1]:.2%})")
            return cached
    except Exception as e:
        logger.error(f"Cache error: {e}")

    logger.info(f"Processing intent for: '{query[:50]}...'")

    # === LOAD KNOWLEDGE BASE ===
    try:
        kb = load_knowledge_base(user_id, chatbot_id=chatbot_id)
    except Exception as e:
        logger.error(f"KB load error: {e}")
        return None, 0.0

    if not kb:
        logger.warning(f"No KB found for chatbot {chatbot_id}")
        return None, 0.0

    # === PREPROCESS ===
    try:
        query_processed = preprocess_text(query)
        st_model = get_sentence_transformer()
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None, 0.0

    best_intent = None
    best_score = 0.0
    best_match_type = None

    # ============================================================
    # PHASE 0: GENERAL INTENTS (Highest Priority)
    # ============================================================
    GENERAL_INTENTS = frozenset([
        'greeting', 'goodbye', 'thanks', 'help', 'about',
        'how_are_you', 'capabilities', 'yes', 'no', 'compliment'
    ])

    try:
        for item in kb:
            intent = item.get('intent') or item.get('tag')
            if not intent or intent.lower() not in GENERAL_INTENTS:
                continue

            patterns = item.get('patterns', [])
            for pattern in patterns:
                pattern_processed = preprocess_text(pattern)

                # Exact match
                if query_processed == pattern_processed:
                    logger.info(f"General exact: {intent} (100%)")
                    result = (intent, 1.0)
                    _safe_cache_set(user_id, cache_key, result, chatbot_id)
                    return result

                # Word overlap (lenient for general intents)
                query_words = set(query_processed.split())
                pattern_words = set(pattern_processed.split())

                if query_words and pattern_words:
                    overlap = len(query_words & pattern_words) / len(query_words)
                    if overlap >= 0.5:
                        confidence = min(0.95, 0.70 + overlap * 0.25)
                        logger.info(f"General match: {intent} ({confidence:.2%})")
                        result = (intent, confidence)
                        _safe_cache_set(user_id, cache_key, result, chatbot_id)
                        return result
    except Exception as e:
        logger.error(f"General intent error: {e}")

        # ============================================================
        # PHASE 1: EXACT & FUZZY MATCH (OPTIMIZED)
        # ============================================================
    try:
        # ✅ OPTIMIZED: Pre-compute query words once
        query_words = set(query_processed.split())
        query_len = len(query_processed)

        for item in kb:
            intent = item.get('intent') or item.get('tag')
            if not intent:
                continue

            patterns = item.get('patterns', [])
            for pattern in patterns:
                pattern_processed = preprocess_text(pattern)

                # ✅ OPTIMIZED: Quick length check before expensive operations
                if abs(len(pattern_processed) - query_len) > query_len * 0.5:
                    continue  # Skip if length difference is too large

                # Exact match (fastest check first)
                if query_processed == pattern_processed:
                    logger.info(f"Exact match: {intent} (100%)")
                    result = (intent, 1.0)
                    _safe_cache_set(user_id, cache_key, result, chatbot_id)
                    return result

                # ✅ OPTIMIZED: Quick word overlap check before fuzzy
                pattern_words = set(pattern_processed.split())
                if query_words and pattern_words:
                    word_overlap = len(query_words & pattern_words) / max(len(query_words), len(pattern_words))
                    if word_overlap >= 0.8:  # High overlap - likely match
                        confidence = min(0.95, 0.70 + word_overlap * 0.25)
                        logger.info(f"High word overlap: {intent} ({confidence:.2%})")
                        result = (intent, confidence)
                        _safe_cache_set(user_id, cache_key, result, chatbot_id)
                        return result

                # Fuzzy match (typo tolerance) - only if above threshold
                if len(query_processed) > 3 and len(pattern_processed) > 3:  # Skip for very short queries
                    fuzzy_score = fuzzy_match_score(query_processed, pattern_processed, max_errors=2)
                    if fuzzy_score >= 0.85:
                        logger.info(f"Fuzzy match: {intent} ({fuzzy_score:.2%})")
                        result = (intent, fuzzy_score)
                        _safe_cache_set(user_id, cache_key, result, chatbot_id)
                        return result
    except Exception as e:
        logger.error(f"Exact/Fuzzy error: {e}")

        # ============================================================
        # PHASE 2: HIGH SEMANTIC SIMILARITY (OPTIMIZED)
        # ============================================================
        # ✅ OPTIMIZED: Only use semantic similarity if no good match yet
        # Semantic similarity is slower, so we skip it if we have a good match
    if st_model and best_score < 0.75:
        try:
            # ✅ OPTIMIZED: Limit semantic search to top candidates
            # Pre-filter by word overlap to reduce expensive semantic calculations
            query_words = set(query_processed.split())
            candidates = []

            for item in kb:
                intent = item.get('intent') or item.get('tag')
                if not intent:
                    continue

                patterns = item.get('patterns', [])
                for pattern in patterns:
                    pattern_words = set(preprocess_text(pattern).split())
                    if query_words and pattern_words:
                        word_overlap = len(query_words & pattern_words) / max(len(query_words), len(pattern_words))
                        if word_overlap > 0.3:  # Only check semantic if some word overlap
                            candidates.append((intent, pattern))

            # ✅ OPTIMIZED: Limit to top 20 candidates for semantic check
            candidates = candidates[:20]

            for intent, pattern in candidates:
                semantic_score = get_semantic_similarity(query, pattern, st_model)

                if semantic_score > 0.70:  # ✅ Slightly higher threshold for better accuracy
                    logger.info(f"Semantic match: {intent} ({semantic_score:.2%})")
                    result = (intent, semantic_score)
                    _safe_cache_set(user_id, cache_key, result, chatbot_id)
                    return result

                # Track best
                if semantic_score > best_score:
                    best_score = semantic_score
                    best_intent = intent
                    best_match_type = "semantic"
        except Exception as e:
            logger.error(f"Semantic error: {e}")

        # ============================================================
        # PHASE 3: ML MODEL (High Confidence) - OPTIMIZED
        # ============================================================
    try:
        # ✅ OPTIMIZED: Only run ML if we don't have a high-confidence match yet
        if best_score < 0.70:  # Only use ML if no strong match found
            ml_intent, ml_confidence = get_ml_intent_prediction(query, user_id, chatbot_id)

            if ml_intent and ml_confidence > 0.50:  # ✅ Slightly higher threshold for better accuracy
                logger.info(f"ML high: {ml_intent} ({ml_confidence:.2%})")
                result = (ml_intent, ml_confidence)
                _safe_cache_set(user_id, cache_key, result, chatbot_id)
                return result

            # Track the best ML prediction
            if ml_intent and ml_confidence > best_score:
                best_score = ml_confidence
                best_intent = ml_intent
                best_match_type = "ml"
    except Exception as e:
        logger.error(f"ML error: {e}")

    # ============================================================
    # PHASE 4: PATTERN MATCHING
    # ============================================================
    try:
        for item in kb:
            intent = item.get('intent') or item.get('tag')
            if not intent:
                continue

            patterns = item.get('patterns', [])
            for pattern in patterns:
                pattern_score = calculate_pattern_similarity(query, pattern)

                if pattern_score > best_score:
                    best_score = pattern_score
                    best_intent = intent
                    best_match_type = "pattern"
    except Exception as e:
        logger.error(f"Pattern error: {e}")

    # ============================================================
    # FINAL DECISION
    # ============================================================
    MIN_CONFIDENCE = 0.30  # Minimum to use KB response

    if best_intent and best_score >= MIN_CONFIDENCE:
        logger.info(f"✓ Matched: '{best_intent}' ({best_score:.2%}) via {best_match_type}")
        result = (best_intent, best_score)
        _safe_cache_set(user_id, cache_key, result, chatbot_id, ttl=1800)
        return result
    else:
        # Low confidence or no match - will use Ollama
        if best_score > 0.0:
            logger.info(f"⚠ Weak match: '{best_intent}' ({best_score:.2%}) - may use Ollama")
            result = (best_intent, best_score)
            _safe_cache_set(user_id, cache_key, result, chatbot_id, ttl=600)
            return result
        else:
            logger.info(f"✗ No match found - will use Ollama for unknown query")
            return None, 0.0


@with_timeout(5)
def get_response_from_kb(intent: str, user_id: int, chatbot_id: int = None) -> Optional[str]:
    """Get response from knowledge base"""

    if not chatbot_id:
        print(f"❌ ERROR: chatbot_id required")
        return None

    kb = load_knowledge_base(user_id, chatbot_id=chatbot_id)

    if not kb:
        return None

    try:
        intent_lower = intent.lower().strip()

        for item in kb:
            item_intent = (item.get('intent') or item.get('tag', '')).lower().strip()

            if item_intent == intent_lower:
                responses = item.get('responses', [])

                if not responses:
                    continue

                for resp in responses:
                    if resp and resp.strip():
                        return clean_html_tags(resp.strip())

        return None

    except Exception as e:
        error_msg = f"❌ KB response error: {e}"
        print(error_msg)
        # Log to file in production
        log_file = os.getenv('LOG_FILE')
        if log_file:
            try:
                import logging
                logging.error(error_msg)
            except:
                pass


# ============================================================================
# OLLAMA
# ============================================================================

def get_available_ollama_model() -> Optional[str]:
    if not OLLAMA_AVAILABLE or ollama is None:
        return None
    try:
        api_response = ollama.list()
        models_data = api_response.get('models', [])
        if not models_data:
            return None
        available = []
        for item in models_data:
            name = None
            if hasattr(item, 'model'):
                name = item.model
            elif isinstance(item, dict):
                name = item.get('model') or item.get('name')
            if name:
                available.append(name.split(':')[0])
        available = list(set(available))
        if not available:
            return None
        for pref in OLLAMA_MODELS:
            if pref in available:
                return pref
        return available[0]
    except Exception:
        return None


def generate_ollama_response(query: str, user_id: int, context: str = None, session_id: str = None) -> Optional[str]:
    """Generate Ollama response"""
    # Check if Ollama is enabled
    if os.getenv('OLLAMA_AVAILABLE', 'False').lower() == 'false':
        return None

    if not OLLAMA_AVAILABLE or ollama is None:
        return None

    timeout = int(os.getenv('OLLAMA_TIMEOUT', '8'))
    try:
        model = get_available_ollama_model()
        if not model:
            return None

        kb_context = ""
        if context:
            kb_context = f"Context: {context}\n\n"

        session_ctx = ""
        if session_id:
            session_ctx = SESSION_MANAGER.get_context(session_id, user_id)
            if session_ctx:
                session_ctx = f"Previous:\n{session_ctx}\n\n"

        system_prompt = """You are a helpful, knowledgeable assistant. 
        Provide clear, accurate, and helpful answers.
        Keep responses concise (2-4 sentences) but informative.
        If you don't know something, say so briefly instead of guessing."""

        # Simple, direct prompt
        user_prompt = f"{kb_context}{session_ctx}Question: {query}\n\nAnswer:"

        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': 0.5,  # Balanced creativity
                'top_p': 0.9,
                'num_predict': 200,  # Allow fuller answers
                'num_ctx': 2048
            },
            stream=False
        )

        answer = response.get('message', {}).get('content', '').strip()

        if not answer or len(answer) < 5:
            return None

        if not validate_ollama_response(answer, query):
            return None

        return answer

    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return None


def validate_ollama_response(response: str, query: str) -> bool:
    if not response or len(response) < 10:
        return False
    generic_responses = [
        "i'm not sure", "i don't have information",
        "i cannot help", "i don't know", "unable to answer"
    ]
    response_lower = response.lower()
    for generic in generic_responses:
        if generic in response_lower:
            return False
    return True


# ============================================================================
# ✅ MAIN RESPONSE FUNCTION
# ============================================================================

def get_smart_response(user_message: str, user_id: int, chatbot_id: int = None,
                       session_id: str = None, is_voice: bool = False) -> str:
    """
    ✅ OPTIMIZED: Smart response with proper intent handling and faster response times
    """

    if not chatbot_id:
        return "Configuration error: Chatbot ID is required."

    # ✅ OPTIMIZED: Faster cache key generation
    cache_key = f"resp_{user_id}_{chatbot_id}_{hashlib.md5(user_message.lower().encode()).hexdigest()}"
    cached = get_cached_response(cache_key)
    if cached:
        print(f"✓ Cache HIT - Instant response")
        return cached

    request_id = REQUEST_TRACKER.start_request(user_id, session_id or "unknown", user_message)
    start_time = time.time()

    # ✅ OPTIMIZED: Ensure model is loaded early (if not already cached)
    # This prevents delay during first prediction
    try:
        ML_MODEL_CACHE.get_model(user_id, chatbot_id, 'svm')
    except:
        pass  # Non-critical, will load later if needed

    try:
        if not RATE_LIMITER.is_allowed(user_id):
            return "Rate limit exceeded. Please wait."

        print(f"\n{'=' * 70}")
        print(f"💬 SMART RESPONSE")
        print(f"   User: {user_id} | Chatbot: {chatbot_id}")
        print(f"   Query: '{user_message[:50]}...'")
        print(f"{'=' * 70}")

        # Intent detection
        intent, confidence = get_intent_with_confidence(user_message, user_id, chatbot_id=chatbot_id)
        print(f"   Intent: '{intent}' | Confidence: {confidence:.1%}")

        response = None
        kb_response_available = False
        ollama_failed = False

        # SEQUENCE 1: HIGH CONFIDENCE (>0.50 - lowered threshold)
        if intent and confidence >= CONFIDENCE_THRESHOLDS['medium']:
            print(f"   🔍 PHASE 1: Intent detected - '{intent}' ({confidence:.2%})")
            print(f"      → Checking KB for response...")

            kb_response = get_response_from_kb(intent, user_id, chatbot_id=chatbot_id)

            if kb_response:
                kb_response_available = True
                response = kb_response
                print(f"      ✓ KB response found - using it!")
                FALLBACK_MANAGER.reset_failure_count(user_id, session_id)

                # Save to session and return immediately
                if session_id:
                    SESSION_MANAGER.add_message(session_id, user_id, user_message, response)
                REQUEST_TRACKER.end_request(request_id, True)
                set_cached_response(cache_key, response)
                print(f"   ⏱ Completed: {time.time() - start_time:.2f}s")
                print(f"{'=' * 70}\n")
                return response
            else:
                print(f"      ℹ Intent exists but no response in KB")
                print(f"      → Will try Ollama for better answer")

        # SEQUENCE 2: MEDIUM CONFIDENCE (0.40-0.65 - lowered threshold)
        if not response:
            if intent and confidence >= CONFIDENCE_THRESHOLDS['medium']:
                # Intent matched but no KB response - try Ollama
                print(f"   🤖 PHASE 2A: Intent matched but no KB response")
                print(f"      → Using Ollama for '{intent}'...")
            elif confidence < CONFIDENCE_THRESHOLDS['medium']:
                # Low confidence or no intent - unknown query
                print(f"   🤖 PHASE 2B: Unknown query (conf={confidence:.2%})")
                print(f"      → Using Ollama for unknown topic...")
            else:
                # No intent at all
                print(f"   🤖 PHASE 2C: No intent detected")
                print(f"      → Using Ollama...")

            if OLLAMA_AVAILABLE:
                ollama_response = generate_ollama_response(
                    user_message, user_id,
                    session_id=session_id
                )

                if ollama_response:
                    response = ollama_response
                    print(f"      ✓ Ollama generated response")
                    FALLBACK_MANAGER.reset_failure_count(user_id, session_id)
                else:
                    ollama_failed = True
                    print(f"      ✗ Ollama failed to generate response")
            else:
                print(f"      ✗ Ollama not available")

        # SEQUENCE 3: OLLAMA (if no good response yet)
        if OLLAMA_AVAILABLE and not response:
            print(f"   🤖 SEQ 3: OLLAMA")
            ollama_response = generate_ollama_response(
                user_message, user_id,
                session_id=session_id
            )

            if ollama_response:
                response = ollama_response
                print(f"      ✓ Ollama response")
                FALLBACK_MANAGER.reset_failure_count(user_id, session_id)
            else:
                ollama_failed = True

        # SEQUENCE 4: SMART FALLBACK
        if not response:
            print(f"   ⚠ SEQ 4: FALLBACK")
            response = FALLBACK_MANAGER.get_smart_fallback(
                confidence=confidence,
                kb_available=kb_response_available,
                ollama_failed=ollama_failed,
                user_id=user_id,
                session_id=session_id,
                is_voice=is_voice
            )

        if session_id and response:
            SESSION_MANAGER.add_message(session_id, user_id, user_message, response)

        REQUEST_TRACKER.end_request(request_id, True)
        print(f"   ⏱ Completed: {time.time() - start_time:.2f}s\n")

        if response and confidence >= CONFIDENCE_THRESHOLDS['medium']:
            set_cached_response(cache_key, response)

        return response

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        REQUEST_TRACKER.end_request(request_id, False)
        return FALLBACK_MANAGER.get_fallback('error', user_id, session_id, 0.0, is_voice)


# ============================================================================
# COMPATIBILITY FUNCTIONS
# ============================================================================

def predict_class(message: str, user_id: int, chatbot_id: int = None) -> str:
    intent, _ = get_intent_with_confidence(message, user_id, chatbot_id)
    return intent or 'unknown'


def get_confidence_score(message: str, user_id: int, chatbot_id: int = None) -> float:
    _, confidence = get_intent_with_confidence(message, user_id, chatbot_id)
    return confidence


def get_response_from_intent(intent: str, user_id: int, chatbot_id: int = None) -> Optional[str]:
    return get_response_from_kb(intent, user_id, chatbot_id)


def load_user_model(user_id: int, chatbot_id: int = None) -> bool:
    """✅ OPTIMIZED: Preload models for faster first response"""
    try:
        if not chatbot_id:
            return False

        base_folder = os.getenv('BASE_DATA_FOLDER', 'data')
        models_dir = os.path.join(base_folder, 'users', f'user_{user_id}',
                                  'chatbots', f'chatbot_{chatbot_id}', 'models')

        if not os.path.exists(models_dir):
            return False

        # ✅ OPTIMIZED: Preload SVM first (most commonly used)
        model_types = ['svm', 'neural_network', 'naive_bayes']
        loaded = False

        for model_type in model_types:
            model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
            vectorizer_path = os.path.join(models_dir, f'{model_type}_vectorizer.pkl')

            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                model, vec, le = ML_MODEL_CACHE.get_model(user_id, chatbot_id, model_type)
                if model and vec:
                    print(f"✓ Preloaded {model_type} model for chatbot {chatbot_id}")
                    loaded = True
                    # Warmup the model with a dummy prediction
                    # This initializes internal structures and improves first real prediction speed
                    try:
                        dummy_query = "test"
                        dummy_vector = vec.transform([dummy_query])
                        if hasattr(model, 'predict_proba'):
                            _ = model.predict_proba(dummy_vector)
                        else:
                            _ = model.predict(dummy_vector)
                        print(f"   Model warmed up for faster inference")
                    except Exception as warmup_error:
                        print(f"   Warmup failed (non-critical): {warmup_error}")
                    break  # Only preload one model (SVM preferred)

        return loaded

    except Exception as e:
        print(f" load_user_model error: {e}")
        return False


def get_model_info(user_id: int, chatbot_id: int = None) -> Dict[str, Any]:
    try:
        if chatbot_id:
            base_folder = os.getenv('BASE_DATA_FOLDER', 'data')
            metadata_path = os.path.join(base_folder, 'users', f'user_{user_id}',
                                         'chatbots', f'chatbot_{chatbot_id}', 'model_metadata.json')

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)

            kb = load_knowledge_base(user_id, chatbot_id=chatbot_id)
            return {
                'user_id': user_id,
                'chatbot_id': chatbot_id,
                'is_trained': len(kb) > 0,
                'kb_size': len(kb),
                'intents': len(kb)
            }
        return {'user_id': user_id, 'is_trained': False}
    except Exception as e:
        print(f" get_model_info error: {e}")
        return {'user_id': user_id, 'is_trained': False}


def get_knowledge_base_info(user_id: int, chatbot_id: int = None) -> Dict[str, Any]:
    try:
        kb = load_knowledge_base(user_id, chatbot_id)
        return {
            'available': len(kb) > 0,
            'total': len(kb),
            'user_id': user_id,
            'chatbot_id': chatbot_id
        }
    except Exception:
        return {'available': False, 'total': 0, 'user_id': user_id}


def get_intent_from_text(text: str, user_id: int, chatbot_id: int = None, threshold: float = 0.0) -> Dict[str, Any]:
    intent, confidence = get_intent_with_confidence(text, user_id, chatbot_id)
    return {
        'intent': intent or 'unknown',
        'confidence': confidence,
        'method': 'hybrid',
        'threshold': threshold,
        'chatbot_id': chatbot_id
    }


def get_system_stats() -> Dict[str, Any]:
    return {
        'active_requests': 0,
        'cache_stats': {},
        'session_count': len(SESSION_MANAGER.sessions) if hasattr(SESSION_MANAGER, 'sessions') else 0,
        'ollama_available': OLLAMA_AVAILABLE,
        'transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
    }


def periodic_cleanup():
    try:
        SMART_CACHE._cleanup_expired()
        print("✓ Periodic cleanup completed")
    except Exception as e:
        print(f"❌ Cleanup error: {e}")


def get_fallback_message(situation: str, user_id: int = None,
                         session_id: str = None, is_voice: bool = False) -> str:
    return FALLBACK_MANAGER.get_fallback(situation, user_id, session_id, 0.0, is_voice)


def reset_user_fallback_count(user_id: int, session_id: str = None):
    FALLBACK_MANAGER.reset_failure_count(user_id, session_id)


# EXPORTS
__all__ = [
    'get_smart_response',
    'get_intent_with_confidence',
    'get_response_from_kb',
    'generate_ollama_response',
    'get_fallback_message',
    'reset_user_fallback_count',
    'FALLBACK_MANAGER',
    'predict_class',
    'get_confidence_score',
    'get_response_from_intent',
    'clear_model_cache',
    'load_user_model',
    'get_model_info',
    'get_knowledge_base_info',
    'get_intent_from_text',
    'get_system_stats',
    'periodic_cleanup',
    'OLLAMA_AVAILABLE',
    'CONFIDENCE_THRESHOLDS',
    'REQUEST_TRACKER',
    'RATE_LIMITER',
    'SESSION_MANAGER',
    'SMART_CACHE',
    'ML_MODEL_CACHE'
]