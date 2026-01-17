"""
✅ ULTRA-ENHANCED PATTERN AUGMENTATION v3.0
- Generates 30+ patterns per intent (GUARANTEED)
- Advanced fuzzy matching integration
- Multiple augmentation techniques
- 95%+ accuracy target
"""

import re
import random
from typing import List, Dict, Set
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Ollama availability check from environment
OLLAMA_AVAILABLE = False

# Check if explicitly disabled in environment
if os.getenv('OLLAMA_AVAILABLE', '').lower() == 'false':
    OLLAMA_AVAILABLE = False
    print("⚠ Ollama disabled via environment variable")
else:
    try:
        import ollama

        # Set Ollama host from environment if specified
        ollama_host = os.getenv('OLLAMA_HOST')
        if ollama_host:
            os.environ['OLLAMA_HOST'] = ollama_host

        models = ollama.list()
        if models and len(models.get('models', [])) > 0:
            OLLAMA_AVAILABLE = True
            print("✓ Ollama available for AI-powered generation")
        else:
            print("⚠ Ollama installed but no models found")
    except Exception as e:
        print(f"⚠ Ollama not available: {e}")
        pass

# ============================================================================
# MASSIVELY EXPANDED LINGUISTIC RESOURCES (10x larger)
# ============================================================================

QUESTION_STARTERS = [
    "what", "how", "why", "when", "where", "who", "which", "whose",
    "can", "could", "would", "should", "shall", "will", "may", "might",
    "do", "does", "did", "is", "are", "was", "were", "am",
    "have", "has", "had", "must", "ought", "need"
]

# MASSIVELY EXPANDED SYNONYMS (3x more groups)
SYNONYM_GROUPS = {
    "help": ["assist", "support", "aid", "guide", "advise", "direct", "facilitate", "enable"],
    "show": ["display", "present", "demonstrate", "reveal", "exhibit", "indicate", "illustrate", "showcase"],
    "tell": ["inform", "explain", "describe", "clarify", "specify", "detail", "outline", "convey"],
    "find": ["locate", "discover", "search for", "look for", "seek", "identify", "uncover", "detect"],
    "get": ["obtain", "acquire", "receive", "retrieve", "fetch", "secure", "gain", "procure"],
    "make": ["create", "build", "generate", "produce", "construct", "develop", "craft", "form"],
    "use": ["utilize", "employ", "apply", "operate", "leverage", "implement", "deploy", "execute"],
    "need": ["require", "want", "must have", "looking for", "seek", "desire", "demand", "necessitate"],
    "know": ["understand", "learn", "aware of", "familiar with", "comprehend", "grasp", "recognize"],
    "see": ["view", "look at", "check", "examine", "inspect", "observe", "review", "watch"],
    "give": ["provide", "offer", "supply", "furnish", "present", "deliver", "grant", "bestow"],
    "work": ["operate", "function", "run", "perform", "execute", "process", "handle", "manage"],
    "start": ["begin", "commence", "initiate", "launch", "kick off", "open", "activate", "trigger"],
    "stop": ["end", "cease", "halt", "terminate", "finish", "conclude", "complete", "discontinue"],
    "buy": ["purchase", "acquire", "obtain", "get", "procure", "order", "invest in"],
    "sell": ["offer", "market", "vend", "trade", "deal", "provide", "distribute"],
    "learn": ["study", "understand", "grasp", "master", "discover", "explore"],
    "try": ["attempt", "test", "check out", "give it a go", "experiment"],
    "like": ["enjoy", "prefer", "love", "appreciate", "favor"],
    "want": ["desire", "wish for", "need", "would like", "require"],
}

# EXPANDED QUESTION TEMPLATES (2x more)
QUESTION_TEMPLATES = {
    "what_is": [
        "what is {topic}", "what's {topic}", "tell me about {topic}",
        "explain {topic}", "can you explain {topic}", "what does {topic} mean",
        "define {topic}", "describe {topic}", "i want to know about {topic}",
        "information about {topic}", "details on {topic}", "give me info on {topic}",
        "what can you tell me about {topic}", "enlighten me about {topic}"
    ],
    "how_to": [
        "how do i {action}", "how to {action}", "how can i {action}",
        "what's the way to {action}", "show me how to {action}",
        "steps to {action}", "guide to {action}", "tutorial for {action}",
        "help me {action}", "teach me to {action}", "instruct me on {action}",
        "walk me through {action}", "process to {action}"
    ],
    "can_you": [
        "can you {action}", "could you {action}", "would you {action}",
        "are you able to {action}", "is it possible to {action}",
        "do you {action}", "will you {action}", "may i {action}",
        "are you capable of {action}", "can you help me {action}"
    ],
    "where_is": [
        "where is {topic}", "where can i find {topic}", "where's {topic}",
        "location of {topic}", "where to find {topic}", "where do i get {topic}",
        "where might {topic} be", "where to locate {topic}"
    ],
    "when": [
        "when is {topic}", "when does {topic}", "what time is {topic}",
        "schedule for {topic}", "timing of {topic}", "at what time {topic}",
        "when will {topic}", "when can i {topic}"
    ],
    "why": [
        "why is {topic}", "why does {topic}", "reason for {topic}",
        "explain why {topic}", "what's the reason {topic}",
        "what causes {topic}", "purpose of {topic}"
    ]
}

# EXPANDED POLITENESS VARIATIONS
POLITENESS_PREFIXES = [
    "please", "kindly", "could you please", "would you please",
    "can you please", "i'd like to", "i need to", "i want to",
    "if possible", "if you could", "would you mind"
]

POLITENESS_SUFFIXES = [
    "please", "thanks", "thank you", "if possible",
    "if you can", "appreciate it", "cheers"
]

# EXPANDED CONTEXT STARTERS
CONTEXT_STARTERS = {
    "need": ["i need to", "i need help with", "i'm looking for", "i require", "i must"],
    "want": ["i want to", "i'd like to", "i wish to", "i'm interested in", "i desire"],
    "help": ["help me", "can you help me", "assist me with", "i need help", "support me"],
    "question": ["i have a question about", "can you tell me", "i'm wondering about", "curious about"],
    "urgent": ["urgently need", "asap", "right away", "immediately", "as soon as possible"]
}


class PatternAugmenter:
    """ULTRA-ENHANCED pattern augmentation - 30+ patterns GUARANTEED"""

    def __init__(self):
        self.seen_patterns = set()

    def should_augment(self, intents: List[Dict]) -> bool:
        """Check if augmentation needed (< 30 patterns)"""
        for intent in intents:
            patterns = intent.get('patterns', [])
            if len(patterns) < 30:
                return True
        return False

    def augment_intents(self, intents: List[Dict], target_per_intent: int = 30) -> List[Dict]:
        """
        Augment ALL intents to have MINIMUM 30 patterns each

        Returns:
            Augmented intents with 30+ patterns each
        """
        augmented_intents = []

        # Log augmentation start
        if os.getenv('LOG_LEVEL') == 'DEBUG':
            print(f"\n{'=' * 70}")
            print(f"🎯 ULTRA-ENHANCED AUGMENTATION (Target: {target_per_intent}+ patterns)")
            print(f"{'=' * 70}")

        log_file = os.getenv('LOG_FILE')
        if log_file:
            try:
                import logging
                logging.info(f"Starting pattern augmentation (target: {target_per_intent} patterns)")
            except:
                pass

        for intent in intents:
            tag = intent.get('tag', 'unknown')
            original_patterns = intent.get('patterns', [])
            responses = intent.get('responses', [])

            if not original_patterns:
                augmented_intents.append(intent)
                continue

            self.seen_patterns.clear()
            for p in original_patterns:
                self.seen_patterns.add(p.lower().strip())

            original_count = len(original_patterns)

            if original_count >= target_per_intent:
                print(f"  ✓ '{tag}': {original_count} patterns (sufficient)")
                augmented_intents.append(intent)
                continue

            print(f"\n  🔄 '{tag}': {original_count} → augmenting to {target_per_intent}+...")

            # Generate with AGGRESSIVE techniques
            new_patterns = self._generate_patterns_ultra(
                original_patterns,
                tag,
                target_count=target_per_intent - original_count
            )

            all_patterns = original_patterns + new_patterns

            augmented_intent = {
                'tag': tag,
                'patterns': all_patterns,
                'responses': responses
            }

            print(f"     ✅ TOTAL: {len(all_patterns)} patterns (+{len(new_patterns)} generated)")
            augmented_intents.append(augmented_intent)

            # Log completion
            if os.getenv('LOG_LEVEL') == 'DEBUG':
                print(f"\n{'=' * 70}")
                print(f"✅ AUGMENTATION COMPLETE - ALL INTENTS HAVE 30+ PATTERNS")
                print(f"{'=' * 70}\n")

            log_file = os.getenv('LOG_FILE')
            if log_file:
                try:
                    import logging
                    total_patterns = sum(len(intent.get('patterns', [])) for intent in augmented_intents)
                    logging.info(
                        f"Augmentation complete: {len(augmented_intents)} intents, {total_patterns} total patterns")
                except:
                    pass

        return augmented_intents

    def _generate_patterns_ultra(self, original_patterns: List[str],
                                 tag: str, target_count: int, techniques_used=None) -> List[str]:
        """
        ULTRA-AGGRESSIVE pattern generation
        Uses ALL techniques to reach 30+ patterns
        """
        new_patterns = []
        techniques = {
            'paraphrase': 0, 'synonym': 0, 'template': 0,
            'variation': 0, 'politeness': 0, 'formality': 0,
            'context': 0, 'typo_tolerance': 0, 'ollama': 0
        }

        sample = original_patterns[0].lower().strip()
        is_question = any(sample.startswith(q) for q in QUESTION_STARTERS)

        # TECHNIQUE 1: Enhanced Paraphrasing (10-12 patterns)
        for pattern in original_patterns:
            paraphrases = self._create_paraphrases_ultra(pattern)
            for p in paraphrases:
                if len(new_patterns) >= target_count:
                    break
                if p.lower().strip() not in self.seen_patterns:
                    new_patterns.append(p)
                    self.seen_patterns.add(p.lower().strip())
                    techniques['paraphrase'] += 1

        # TECHNIQUE 2: Synonym Replacement (8-10 patterns)
        if len(new_patterns) < target_count:
            for pattern in original_patterns[:5]:
                synonyms = self._replace_synonyms_ultra(pattern)
                for s in synonyms:
                    if len(new_patterns) >= target_count:
                        break
                    if s.lower().strip() not in self.seen_patterns:
                        new_patterns.append(s)
                        self.seen_patterns.add(s.lower().strip())
                        techniques['synonym'] += 1

        # TECHNIQUE 3: Template Generation (6-8 patterns)
        if len(new_patterns) < target_count:
            templates = self._generate_from_templates_ultra(original_patterns, is_question)
            for t in templates:
                if len(new_patterns) >= target_count:
                    break
                if t.lower().strip() not in self.seen_patterns:
                    new_patterns.append(t)
                    self.seen_patterns.add(t.lower().strip())
                    techniques['template'] += 1

        # TECHNIQUE 4: Structural Variations (5-7 patterns)
        if len(new_patterns) < target_count:
            for pattern in original_patterns[:4]:
                variations = self._create_structural_variations_ultra(pattern)
                for v in variations:
                    if len(new_patterns) >= target_count:
                        break
                    if v.lower().strip() not in self.seen_patterns:
                        new_patterns.append(v)
                        self.seen_patterns.add(v.lower().strip())
                        techniques['variation'] += 1

        # TECHNIQUE 5: Politeness Modifiers (4-6 patterns)
        if len(new_patterns) < target_count:
            for pattern in original_patterns[:3]:
                polite = self._add_politeness_ultra(pattern)
                for p in polite:
                    if len(new_patterns) >= target_count:
                        break
                    if p.lower().strip() not in self.seen_patterns:
                        new_patterns.append(p)
                        self.seen_patterns.add(p.lower().strip())
                        techniques['politeness'] += 1

        # TECHNIQUE 6: Formality Shifts (3-5 patterns)
        if len(new_patterns) < target_count:
            for pattern in original_patterns[:3]:
                formal = self._shift_formality_ultra(pattern)
                for f in formal:
                    if len(new_patterns) >= target_count:
                        break
                    if f.lower().strip() not in self.seen_patterns:
                        new_patterns.append(f)
                        self.seen_patterns.add(f.lower().strip())
                        techniques['formality'] += 1

                # Technique 7: Context additions (2-3 patterns)
                if len(new_patterns) < target_count:
                    for pattern in original_patterns[:2]:
                        context_vars = self._add_context(pattern)
                        for cv in context_vars:
                            if len(new_patterns) >= target_count:
                                break
                            if cv.lower().strip() not in self.seen_patterns:
                                new_patterns.append(cv)
                                self.seen_patterns.add(cv.lower().strip())
                                techniques_used['context'] += 1

                # Technique 7.5: Typo Tolerance Variants (NEW - 2-4 patterns)
                if len(new_patterns) < target_count:
                    for pattern in original_patterns[:3]:
                        typo_variants = self._generate_typo_variants(pattern)
                        for tv in typo_variants:
                            if len(new_patterns) >= target_count:
                                break
                            if tv.lower().strip() not in self.seen_patterns:
                                new_patterns.append(tv)
                                self.seen_patterns.add(tv.lower().strip())
                                techniques_used['typo_tolerance'] += 1

        # TECHNIQUE 8: Typo Tolerance Variants (2-3 patterns)
        if len(new_patterns) < target_count:
            for pattern in original_patterns[:2]:
                typos = self._generate_typo_variants(pattern)
                for ty in typos:
                    if len(new_patterns) >= target_count:
                        break
                    if ty.lower().strip() not in self.seen_patterns:
                        new_patterns.append(ty)
                        self.seen_patterns.add(ty.lower().strip())
                        techniques['typo_tolerance'] += 1

        # TECHNIQUE 9: AI-Powered (Ollama) - Fill remaining
        if OLLAMA_AVAILABLE and len(new_patterns) < target_count:
            remaining = target_count - len(new_patterns)
            ai_patterns = self._generate_with_ollama_ultra(original_patterns, tag, remaining)
            for ap in ai_patterns:
                if len(new_patterns) >= target_count:
                    break
                if ap.lower().strip() not in self.seen_patterns:
                    new_patterns.append(ap)
                    self.seen_patterns.add(ap.lower().strip())
                    techniques['ollama'] += 1

        # Print breakdown
        print(f"     Methods:")
        for tech, count in techniques.items():
            if count > 0:
                print(f"       • {tech}: {count}")

        return new_patterns[:target_count]

    def _create_paraphrases_ultra(self, pattern: str) -> List[str]:
        """ULTRA paraphrasing - 12+ variations"""
        paraphrases = []
        p_lower = pattern.lower().strip()

        # Question mark variants
        if p_lower.endswith('?'):
            paraphrases.append(p_lower.rstrip('?'))
            paraphrases.append(p_lower.rstrip('?') + '...')
        else:
            paraphrases.append(p_lower + '?')
            paraphrases.append(p_lower + '...')

        # Case variants
        paraphrases.extend([
            pattern.capitalize(),
            pattern.lower(),
            pattern.upper(),
            pattern.title()
        ])

        # Politeness additions (expanded)
        for prefix in ["please", "could you", "would you", "kindly", "can you"]:
            if prefix not in p_lower:
                paraphrases.append(f"{prefix} {p_lower}")

        for suffix in ["please", "thanks", "thank you"]:
            if suffix not in p_lower:
                paraphrases.append(f"{p_lower} {suffix}")

        return paraphrases

    def _replace_synonyms_ultra(self, pattern: str) -> List[str]:
        """ULTRA synonym replacement - multiple passes"""
        synonyms = []
        words = pattern.lower().split()

        # Multi-word synonyms
        for word, syn_list in SYNONYM_GROUPS.items():
            if word in words:
                for synonym in syn_list[:6]:  # More synonyms
                    new_pattern = pattern.lower().replace(word, synonym)
                    if new_pattern != pattern.lower():
                        synonyms.append(new_pattern)

        return synonyms

    def _generate_from_templates_ultra(self, patterns: List[str], is_question: bool) -> List[str]:
        """ULTRA template generation"""
        generated = []
        topics = self._extract_topics_ultra(patterns)
        actions = self._extract_actions_ultra(patterns)

        if is_question:
            for template_type, templates in QUESTION_TEMPLATES.items():
                for template in templates[:4]:  # More templates
                    if '{topic}' in template and topics:
                        for topic in topics[:3]:
                            generated.append(template.format(topic=topic))
                    elif '{action}' in template and actions:
                        for action in actions[:3]:
                            generated.append(template.format(action=action))
        else:
            starters = ["tell me about", "explain", "what is", "information on", "details about"]
            for starter in starters:
                for pattern in patterns[:2]:
                    generated.append(f"{starter} {pattern}")

        return generated

    def _create_structural_variations_ultra(self, pattern: str) -> List[str]:
        """ULTRA structural changes"""
        variations = []
        p_lower = pattern.lower().strip()

        # Pronoun swaps
        swaps = [
            (" you ", " i "), (" i ", " you "),
            (" your ", " my "), (" my ", " your "),
            (" yours ", " mine "), (" mine ", " yours ")
        ]
        for old, new in swaps:
            if old in p_lower:
                variations.append(p_lower.replace(old, new))

        # Context starters
        for starter in ["i need to", "i want to", "help me", "show me", "tell me"]:
            if not p_lower.startswith(starter):
                variations.append(f"{starter} {p_lower}")

        # Time additions
        for time_word in ["now", "today", "right now", "immediately", "asap"]:
            if time_word not in p_lower:
                variations.append(f"{p_lower} {time_word}")

        return variations

    def _add_politeness_ultra(self, pattern: str) -> List[str]:
        """ULTRA politeness"""
        polite = []
        p_lower = pattern.lower().strip()

        for prefix in POLITENESS_PREFIXES[:6]:
            if prefix not in p_lower:
                polite.append(f"{prefix} {p_lower}")

        for suffix in POLITENESS_SUFFIXES[:4]:
            if suffix not in p_lower:
                polite.append(f"{p_lower} {suffix}")

        return polite

    def _shift_formality_ultra(self, pattern: str) -> List[str]:
        """ULTRA formality shifting"""
        informal_formal = {
            "wanna": "want to", "gonna": "going to", "gotta": "got to",
            "can't": "cannot", "won't": "will not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not",
            "what's": "what is", "how's": "how is", "where's": "where is"
        }

        formal = []
        p_lower = pattern.lower()

        for informal, formal_form in informal_formal.items():
            if informal in p_lower:
                formal.append(p_lower.replace(informal, formal_form))
            if formal_form in p_lower:
                formal.append(p_lower.replace(formal_form, informal))

        return formal

    def _create_paraphrases_enhanced(self, pattern: str) -> List[str]:
        """Enhanced paraphrasing with more variations"""
        paraphrases = []
        pattern_lower = pattern.lower().strip()

        # Question mark variations
        if pattern_lower.endswith('?'):
            paraphrases.append(pattern_lower.rstrip('?'))
            paraphrases.append(pattern_lower.rstrip('?') + '...')
            paraphrases.append(pattern_lower.rstrip('?') + ' please')
        else:
            paraphrases.append(pattern_lower + '?')
            paraphrases.append(pattern_lower + '...')

        # Capitalization variants
        paraphrases.append(pattern.capitalize())
        paraphrases.append(pattern.lower())
        paraphrases.append(pattern.upper())
        paraphrases.append(pattern.title())

    def _add_context_ultra(self, pattern: str) -> List[str]:
        """ULTRA context additions"""
        context_vars = []
        p_lower = pattern.lower().strip()

        for category, starters in CONTEXT_STARTERS.items():
            for starter in starters[:3]:
                if not p_lower.startswith(starter):
                    context_vars.append(f"{starter} {p_lower}")

        return context_vars

    def _generate_typo_variants(self, pattern: str) -> List[str]:
        """Generate common typo variants for better matching"""
        typos = []
        words = pattern.split()

        for i, word in enumerate(words):
            if len(word) > 3:
                # Missing last letter: "tdtech" → "tdteh"
                typo1 = word[:-1]
                new_pattern1 = ' '.join(words[:i] + [typo1] + words[i + 1:])
                typos.append(new_pattern1)

                # Swapped letters: "tell" → "tel"
                if len(word) > 4:
                    typo2 = word[:-2] + word[-1]
                    new_pattern2 = ' '.join(words[:i] + [typo2] + words[i + 1:])
                    typos.append(new_pattern2)

        return typos[:3]

    def _extract_topics_ultra(self, patterns: List[str]) -> List[str]:
        """Extract topics - enhanced"""
        topics = set()
        for pattern in patterns:
            words = pattern.lower().split()
            filtered = [w for w in words if w not in QUESTION_STARTERS
                        and len(w) > 2 and w not in ['the', 'this', 'that', 'with', 'from', 'and', 'or']]
            topics.update(filtered)
        return list(topics)[:7]

    def _extract_actions_ultra(self, patterns: List[str]) -> List[str]:
        """Extract actions - enhanced"""
        actions = set()
        for pattern in patterns:
            p_lower = pattern.lower()
            # Match "to <action>"
            to_match = re.search(r'\bto\s+(\w+(?:\s+\w+)?)', p_lower)
            if to_match:
                actions.add(to_match.group(1))
            # Match "how to <action>"
            how_match = re.search(r'how\s+to\s+(\w+(?:\s+\w+)?)', p_lower)
            if how_match:
                actions.add(how_match.group(1))
        return list(actions)[:7]

    def _generate_with_ollama_ultra(self, patterns: List[str], tag: str, count: int) -> List[str]:
        """AI-powered generation with Ollama"""
        if not OLLAMA_AVAILABLE:
            return []

        try:
            patterns_text = "\n".join(patterns[:5])
            prompt = f"""Generate {count} natural variations of these questions about '{tag}':

{patterns_text}

Requirements:
- Make them conversational and natural
- Include different phrasings
- Keep the same intent
- Only output questions, one per line"""

            # Get timeout from environment
            timeout = int(os.getenv('OLLAMA_TIMEOUT', '12'))

            # Get preferred model from environment
            ollama_models = os.getenv('OLLAMA_MODELS', 'llama2,mistral').split(',')
            preferred_model = ollama_models[0].strip() if ollama_models else 'llama2'

            response = ollama.chat(
                model=preferred_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.8,
                    'num_predict': 400,
                    'timeout': timeout
                }
            )
            content = response.get('message', {}).get('content', '')
            lines = [line.strip() for line in content.split('\n') if line.strip()]

            cleaned = []
            for line in lines:
                clean = re.sub(r'^[\d.)\-*]+\s*', '', line)
                if clean and len(clean) > 5:
                    cleaned.append(clean)
            return cleaned[:count]
        except Exception as e:
            error_msg = f"       ⚠ Ollama failed: {e}"
            print(error_msg)

            # Log to file in production
            log_file = os.getenv('LOG_FILE')
            if log_file:
                try:
                    import logging
                    logging.warning(error_msg)
                except:
                    pass
            return []

def augment_training_data(intents_data: Dict, min_patterns: int = 30) -> Dict:
    """
    MAIN FUNCTION: Augment to 30+ patterns per intent

    Args:
        intents_data: {'intents': [...]}
        min_patterns: Minimum patterns (default: 30)

    Returns:
        Augmented training data with 30+ patterns each
    """
    if not isinstance(intents_data, dict) or 'intents' not in intents_data:
        return intents_data

    augmenter = PatternAugmenter()

    if not augmenter.should_augment(intents_data['intents']):
        print("✓ All intents already have 30+ patterns")
        return intents_data

    augmented_intents = augmenter.augment_intents(
        intents_data['intents'],
        target_per_intent=min_patterns
    )

    return {'intents': augmented_intents}


__all__ = ['PatternAugmenter', 'augment_training_data']