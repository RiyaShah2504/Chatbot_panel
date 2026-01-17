# database.py - ENHANCED VERSION with Conversation Analytics
import json
from datetime import datetime, timezone, timedelta

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


#User Table
class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    subscription = db.relationship('Subscription', backref='user', uselist=False, cascade='all, delete-orphan')
    chatbots = db.relationship('Chatbot', backref='owner', lazy=True, cascade='all, delete-orphan')


#Subscription Plans Table
class SubscriptionPlan(db.Model):
    __tablename__ = 'subscription_plans'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    billing_cycle = db.Column(db.String(20), default='monthly')
    max_chatbots = db.Column(db.Integer, default=1)
    max_messages_per_month = db.Column(db.Integer, default=100)
    max_training_data_size = db.Column(db.Integer, default=1000)
    features = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<SubscriptionPlan {self.display_name}>'


#User Subscription Table
class Subscription(db.Model):
    __tablename__ = 'subscriptions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('subscription_plans.id'), nullable=False)
    status = db.Column(db.String(20), default='active')
    start_date = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    end_date = db.Column(db.DateTime(timezone=True))
    trial_end_date = db.Column(db.DateTime(timezone=True))
    is_trial = db.Column(db.Boolean, default=False)
    chatbots_created = db.Column(db.Integer, default=0)
    messages_this_month = db.Column(db.Integer, default=0)
    last_reset_date = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    payment_method = db.Column(db.String(50))
    last_payment_date = db.Column(db.DateTime(timezone=True))
    next_billing_date = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
    plan = db.relationship('SubscriptionPlan', backref='subscriptions')

    def is_expired(self):
        """Check if subscription is expired"""
        now = datetime.now(timezone.utc)
        def _to_aware(dt):
            if dt and dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        if self.is_trial and self.trial_end_date:
            trial_end = _to_aware(self.trial_end_date)
            return now > trial_end

        if self.end_date:
            end = _to_aware(self.end_date)
            return now > end

        return False

    def days_remaining(self):
        """Get days remaining in subscription"""
        now = datetime.now(timezone.utc)

        def _to_aware(dt):
            if dt and dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        target_date = None
        if self.is_trial and self.trial_end_date:
            target_date = _to_aware(self.trial_end_date)
        elif self.end_date:
            target_date = _to_aware(self.end_date)

        if target_date:
            delta = target_date - now
            return max(0, delta.days)
        return 0

    def can_send_message(self):
        """Check if user can send another message"""
        if self.plan.max_messages_per_month == -1:
            return True
        return self.messages_this_month < self.plan.max_messages_per_month

    def can_create_chatbot(self):
        """Check if user can create another chatbot"""
        if self.plan.max_chatbots == -1:
            return True
        return self.chatbots_created < self.plan.max_chatbots

    def increment_chatbot_count(self):
        """Increment chatbot counter"""
        self.chatbots_created += 1
        self.updated_at = datetime.now(timezone.utc)
        db.session.commit()

    def decrement_chatbot_count(self):
        """Decrement chatbot counter"""
        if self.chatbots_created > 0:
            self.chatbots_created -= 1
            self.updated_at = datetime.now(timezone.utc)
            db.session.commit()

    def increment_message_count(self):
        """Increment message counter"""
        self.messages_this_month += 1
        self.updated_at = datetime.now(timezone.utc)
        db.session.commit()

    def reset_monthly_counters(self):
        """Reset monthly counters"""
        self.messages_this_month = 0
        self.last_reset_date = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        db.session.commit()

    def get_usage_stats(self):
        """Get usage statistics"""
        chatbot_limit = self.plan.max_chatbots
        message_limit = self.plan.max_messages_per_month

        return {
            'chatbots': {
                'used': self.chatbots_created,
                'limit': 'Unlimited' if chatbot_limit == -1 else chatbot_limit,
                'remaining': 'Unlimited' if chatbot_limit == -1 else max(0, chatbot_limit - self.chatbots_created),
                'percentage': 0 if chatbot_limit == -1 else min(100, (self.chatbots_created / chatbot_limit) * 100)
            },
            'messages': {
                'used': self.messages_this_month,
                'limit': 'Unlimited' if message_limit == -1 else message_limit,
                'remaining': 'Unlimited' if message_limit == -1 else max(0, message_limit - self.messages_this_month),
                'percentage': 0 if message_limit == -1 else min(100, (self.messages_this_month / message_limit) * 100)
            },
            'subscription': {
                'days_remaining': self.days_remaining(),
                'is_expired': self.is_expired(),
                'is_trial': self.is_trial,
                'status': self.status,
                'plan_name': self.plan.display_name
            }
        }

    def __repr__(self):
        return f'<Subscription user_id={self.user_id} plan={self.plan.name} status={self.status}>'


class Chatbot(db.Model):
    __tablename__ = 'chatbot'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    embed_code = db.Column(db.String(500), unique=True)
    is_active = db.Column(db.Boolean, default=False)
    training_data = db.Column(db.Text)
    training_file = db.Column(db.String(200))
    theme_color = db.Column(db.String(7), default='#4F46E5')
    welcome_message = db.Column(db.String(200), default='Hello! How can I help you?')
    bot_name = db.Column(db.String(100), default='AI Assistant')
    use_ml_model = db.Column(db.Boolean, default=False)
    intents_path = db.Column(db.String(255))
    trained_folder = db.Column(db.String(255))
    is_trained = db.Column(db.Boolean, default=False)

    #Avatar and styling
    bot_avatar = db.Column(db.String(500))
    welcome_button_text = db.Column(db.String(100))
    welcome_button_url = db.Column(db.String(500))
    chat_background_color = db.Column(db.String(7), default='#F7FAFC')
    user_message_color = db.Column(db.String(7))
    bot_message_color = db.Column(db.String(7), default='#FFFFFF')
    user_text_color = db.Column(db.String(7), default='#FFFFFF')
    bot_text_color = db.Column(db.String(7), default='#1A202C')

    #Welcome buttons with submenu support
    welcome_buttons = db.Column(db.Text, comment='JSON data for welcome buttons with submenu support')

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def get_welcome_buttons_dict(self):
        """Parse and return welcome_buttons as a Python list"""
        if not self.welcome_buttons:
            return []

        try:
            buttons = json.loads(self.welcome_buttons)
            return buttons if isinstance(buttons, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    def set_welcome_buttons_dict(self, buttons_list):
        """Set welcome_buttons from a Python list"""
        if not isinstance(buttons_list, list):
            self.welcome_buttons = '[]'
            return False

        valid_types = ['url', 'intent', 'message', 'submenu']
        validated_buttons = []

        for button in buttons_list:
            if not isinstance(button, dict):
                continue

            button_data = {
                'text': button.get('text', '').strip(),
                'type': button.get('type', 'url').strip(),
                'value': button.get('value', '').strip(),
                'has_submenu': button.get('has_submenu', False),
                'submenu_items': []
            }

            if button_data['type'] not in valid_types:
                button_data['type'] = 'url'

            if button_data['has_submenu'] and 'submenu_items' in button:
                submenu_items = button.get('submenu_items', [])
                if isinstance(submenu_items, list):
                    for sub_item in submenu_items:
                        if isinstance(sub_item, dict) and sub_item.get('text', '').strip():
                            sub_type = sub_item.get('type', 'url').strip()
                            if sub_type not in valid_types:
                                sub_type = 'url'

                            button_data['submenu_items'].append({
                                'text': sub_item.get('text', '').strip(),
                                'type': sub_type,
                                'value': sub_item.get('value', '').strip()
                            })

            if button_data['text']:
                validated_buttons.append(button_data)

        self.welcome_buttons = json.dumps(validated_buttons)
        return True

    def preprocess(self, text: str) -> str:
        """Dynamic preprocessing"""
        import re
        if not text:
            return ""

        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        if self.training_file and 'faq' in (self.training_file or "").lower():
            text = text.replace('?', '')

        return text

    def __repr__(self):
        return f'<Chatbot {self.name} (user_id={self.user_id})>'


#Chat Session Table with Analytics
class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'

    id = db.Column(db.Integer, primary_key=True)
    chatbot_id = db.Column(db.Integer, db.ForeignKey('chatbot.id'), nullable=True)
    session_token = db.Column(db.String(100), unique=True)  # NEW: For tracking

    #User identification
    user_name = db.Column(db.String(100))
    user_email = db.Column(db.String(120))
    user_ip = db.Column(db.String(50))  # NEW: Track IP for analytics
    user_agent = db.Column(db.Text)  # NEW: Track browser/device

    #Session timing
    started_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ended_at = db.Column(db.DateTime(timezone=True))
    last_activity = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    #Analytics
    message_count = db.Column(db.Integer, default=0)  # NEW
    avg_confidence = db.Column(db.Float, default=0.0)  # NEW
    fallback_count = db.Column(db.Integer, default=0)  # NEW
    satisfaction_rating = db.Column(db.Integer)  # NEW: 1-5 rating

    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
        db.session.commit()

    def increment_message_count(self):
        """Increment message counter"""
        self.message_count += 1
        self.update_activity()

    def is_active(self, timeout_minutes=30):
        """Check if session is still active"""
        if not self.last_activity:
            return False

        now = datetime.now(timezone.utc)
        if self.last_activity.tzinfo is None:
            last_activity = self.last_activity.replace(tzinfo=timezone.utc)
        else:
            last_activity = self.last_activity

        return (now - last_activity).total_seconds() < (timeout_minutes * 60)

    def get_duration(self):
        """Get session duration in seconds"""
        if not self.started_at:
            return 0

        end = self.ended_at if self.ended_at else datetime.now(timezone.utc)

        start = self.started_at
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        return (end - start).total_seconds()

    def __repr__(self):
        return f'<ChatSession {self.id} for Chatbot {self.chatbot_id}>'


#Chat Message Table
class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)

    #Message content
    sender = db.Column(db.String(10))  # 'user' or 'bot'
    message = db.Column(db.Text)

    #AI Analytics
    intent = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    is_fallback = db.Column(db.Boolean, default=False)
    processing_time_ms = db.Column(db.Integer)  # NEW: Response time

    #Context
    extra_data = db.Column(db.Text)  # JSON data for additional context
    timestamp = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'sender': self.sender,
            'message': self.message,
            'intent': self.intent,
            'confidence': self.confidence,
            'is_fallback': self.is_fallback,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

    def __repr__(self):
        return f'<ChatMessage {self.id} from {self.sender}>'


#QA Pair Table
class QAPair(db.Model):
    __tablename__ = 'qa_pairs'

    id = db.Column(db.Integer, primary_key=True)
    chatbot_id = db.Column(db.Integer, db.ForeignKey('chatbot.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    tag = db.Column(db.String(100))

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    chatbot = db.relationship('Chatbot', backref=db.backref('qa_pairs', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        """Convert QA pair to dictionary"""

        def format_datetime(dt):
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()

        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'tag': self.tag,
            'created_at': format_datetime(self.created_at),
            'updated_at': format_datetime(self.updated_at)
        }

    def __repr__(self):
        return f'<QAPair {self.id} - Chatbot {self.chatbot_id}>'



# HELPER FUNCTIONS
def save_qa_pairs_to_db(chatbot_id, qa_data_list):
    """Save or update QA pairs"""
    try:
        QAPair.query.filter_by(chatbot_id=chatbot_id).delete()
        current_time = datetime.now(timezone.utc)

        for idx, qa_data in enumerate(qa_data_list):
            qa_pair = QAPair(
                chatbot_id=chatbot_id,
                question=qa_data['question'],
                answer=qa_data['answer'],
                tag=qa_data.get('tag', f'qa_{idx + 1}'),
                created_at=current_time,
                updated_at=current_time
            )
            db.session.add(qa_pair)

        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        print(f"Error saving QA pairs: {e}")
        return False


def get_qa_pairs_from_db(chatbot_id):
    """Get all QA pairs"""
    qa_pairs = QAPair.query.filter_by(chatbot_id=chatbot_id).order_by(QAPair.created_at).all()
    return [qa.to_dict() for qa in qa_pairs]


def update_single_qa_pair(qa_id, question, answer):
    """Update a single QA pair"""
    try:
        qa_pair = QAPair.query.get(qa_id)
        if qa_pair:
            qa_pair.question = question
            qa_pair.answer = answer
            qa_pair.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            return True
        return False
    except Exception as e:
        db.session.rollback()
        print(f"Error updating QA pair: {e}")
        return False


def delete_qa_pair(qa_id):
    """Delete a single QA pair"""
    try:
        qa_pair = QAPair.query.get(qa_id)
        if qa_pair:
            db.session.delete(qa_pair)
            db.session.commit()
            return True
        return False
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting QA pair: {e}")
        return False


#Chat Session Helper Functions
def create_session(chatbot_id=None, name=None, email=None, ip=None, user_agent=None):
    """Create a new chat session with analytics"""
    import secrets

    session_token = secrets.token_urlsafe(32)

    new_session = ChatSession(
        chatbot_id=chatbot_id,
        session_token=session_token,
        user_name=name,
        user_email=email,
        user_ip=ip,
        user_agent=user_agent
    )
    db.session.add(new_session)
    db.session.commit()
    return new_session.id, session_token


def log_message(session_id, sender, message, intent=None, confidence=None,
                is_fallback=False, processing_time=None, extra=None):
    """Log a chat message with analytics"""
    msg = ChatMessage(
        session_id=session_id,
        sender=sender,
        message=message,
        intent=intent,
        confidence=confidence,
        is_fallback=is_fallback,
        processing_time_ms=processing_time,
        extra_data=json.dumps(extra or {})
    )
    db.session.add(msg)

    #Update session analytics
    session = ChatSession.query.get(session_id)
    if session:
        session.increment_message_count()

        if is_fallback:
            session.fallback_count = (session.fallback_count or 0) + 1

        # Update average confidence
        if confidence and sender == 'bot':
            if session.avg_confidence == 0:
                session.avg_confidence = confidence
            else:
                # Running average
                session.avg_confidence = (session.avg_confidence + confidence) / 2

    db.session.commit()
    return msg.id


def end_session(session_id):
    """End a chat session"""
    session = ChatSession.query.get(session_id)
    if session:
        session.ended_at = datetime.now(timezone.utc)
        db.session.commit()
        return True
    return False


def get_session_messages(session_id):
    """Get all messages from a session"""
    return ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()


def get_chatbot_sessions(chatbot_id, limit=50):
    """Get recent sessions for a chatbot"""
    return ChatSession.query.filter_by(chatbot_id=chatbot_id).order_by(
        ChatSession.started_at.desc()
    ).limit(limit).all()


def get_chatbot_analytics(chatbot_id, days=30):
    """Get analytics for a chatbot"""

    since = datetime.now(timezone.utc) - timedelta(days=days)

    sessions = ChatSession.query.filter(
        ChatSession.chatbot_id == chatbot_id,
        ChatSession.started_at >= since
    ).all()

    total_sessions = len(sessions)
    total_messages = sum(s.message_count for s in sessions)
    avg_confidence = sum(s.avg_confidence for s in sessions if s.avg_confidence) / max(1, total_sessions)
    total_fallbacks = sum(s.fallback_count or 0 for s in sessions)

    return {
        'total_sessions': total_sessions,
        'total_messages': total_messages,
        'avg_messages_per_session': total_messages / max(1, total_sessions),
        'avg_confidence': avg_confidence,
        'total_fallbacks': total_fallbacks,
        'fallback_rate': total_fallbacks / max(1, total_messages)
    }


#Subscription Helper Functions
def initialize_subscription_plans():
    """Initialize default subscription plans"""
    plans = [
        {
            'name': 'free_trial',
            'display_name': 'Free Trial',
            'price': 0.0,
            'billing_cycle': 'trial',
            'max_chatbots': 1,
            'max_messages_per_month': 100,
            'max_training_data_size': 500,
            'features': json.dumps({
                'chatbots': 1,
                'messages': 100,
                'ml_training': False,
                'custom_branding': False,
                'analytics': 'basic',
                'support': 'email',
                'api_access': False,
                'priority_support': False
            })
        },
        {
            'name': 'basic',
            'display_name': 'Basic',
            'price': 9.99,
            'billing_cycle': 'monthly',
            'max_chatbots': 3,
            'max_messages_per_month': 1000,
            'max_training_data_size': 2000,
            'features': json.dumps({
                'chatbots': 3,
                'messages': 1000,
                'ml_training': True,
                'custom_branding': False,
                'analytics': 'basic',
                'support': 'email',
                'api_access': False,
                'priority_support': False
            })
        },
        {
            'name': 'moderate',
            'display_name': 'Moderate',
            'price': 29.99,
            'billing_cycle': 'monthly',
            'max_chatbots': 10,
            'max_messages_per_month': 5000,
            'max_training_data_size': 10000,
            'features': json.dumps({
                'chatbots': 10,
                'messages': 5000,
                'ml_training': True,
                'custom_branding': True,
                'analytics': 'advanced',
                'support': 'priority',
                'api_access': True,
                'priority_support': False
            })
        },
        {
            'name': 'advanced',
            'display_name': 'Advanced',
            'price': 99.99,
            'billing_cycle': 'monthly',
            'max_chatbots': -1,
            'max_messages_per_month': -1,
            'max_training_data_size': -1,
            'features': json.dumps({
                'chatbots': 'unlimited',
                'messages': 'unlimited',
                'ml_training': True,
                'custom_branding': True,
                'analytics': 'advanced',
                'support': 'dedicated',
                'api_access': True,
                'priority_support': True,
                'white_label': True
            })
        }
    ]

    for plan_data in plans:
        existing = SubscriptionPlan.query.filter_by(name=plan_data['name']).first()
        if not existing:
            plan = SubscriptionPlan(**plan_data)
            db.session.add(plan)

    db.session.commit()
    print(f" Initialized {len(plans)} subscription plans")


def create_trial_subscription(user_id):
    """Create trial subscription"""
    trial_plan = SubscriptionPlan.query.filter_by(name='free_trial').first()
    if not trial_plan:
        print(" Error: Free trial plan not found!")
        return None

    trial_end = datetime.now(timezone.utc) + timedelta(days=14)

    subscription = Subscription(
        user_id=user_id,
        plan_id=trial_plan.id,
        status='trial',
        is_trial=True,
        trial_end_date=trial_end,
        end_date=trial_end
    )

    db.session.add(subscription)
    db.session.commit()
    print(f" Created trial subscription for user {user_id}")
    return subscription


def get_subscription_by_user_id(user_id):
    """Get subscription for user"""
    return Subscription.query.filter_by(user_id=user_id).first()


def update_subscription_plan(user_id, new_plan_name):
    """Update subscription plan"""
    subscription = get_subscription_by_user_id(user_id)
    if not subscription:
        print(f" No subscription found for user {user_id}")
        return False

    new_plan = SubscriptionPlan.query.filter_by(name=new_plan_name).first()
    if not new_plan:
        print(f" Plan '{new_plan_name}' not found")
        return False

    subscription.plan_id = new_plan.id
    subscription.status = 'active'
    subscription.is_trial = (new_plan_name == 'free_trial')
    subscription.updated_at = datetime.now(timezone.utc)

    if new_plan_name == 'free_trial':
        trial_end = datetime.now(timezone.utc) + timedelta(days=14)
        subscription.trial_end_date = trial_end
        subscription.end_date = trial_end
    else:
        subscription.end_date = datetime.now(timezone.utc) + timedelta(days=30)
        subscription.next_billing_date = subscription.end_date

    db.session.commit()
    print(f" Updated subscription for user {user_id} to {new_plan.display_name}")
    return True


def reset_all_monthly_counters():
    """Reset monthly counters"""
    subscriptions = Subscription.query.filter_by(status='active').all()
    count = 0
    for subscription in subscriptions:
        subscription.reset_monthly_counters()
        count += 1
    print(f" Reset {count} subscription message counters")
    return count


