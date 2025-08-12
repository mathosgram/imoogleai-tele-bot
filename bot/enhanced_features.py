"""
Enhanced Features for Multi-Provider AI Telegram Bot
====================================================

This module adds amazing new features to make your bot more powerful:
- Smart response mode switching
- Auto-model optimization
- Enhanced conversation context
- Dynamic personality adaptation
- Multi-language intelligence
- Cost optimization
- Response quality scoring
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re


class EnhancedBot:
    """Enhanced bot features for better user experience"""
    
    def __init__(self, ai_helper, config: dict):
        self.ai_helper = ai_helper
        self.config = config
        self.conversation_analytics = {}
        self.user_preferences = {}
        self.response_cache = {}
        self.auto_optimization = config.get('AUTO_OPTIMIZATION', True)
        
    def get_smart_model_suggestion(self, query: str, chat_id: int) -> Optional[str]:
        """Suggest the best model based on query type and user history"""
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # For coding tasks
        if any(keyword in query_lower for keyword in ['code', 'programming', 'debug', 'function', 'script', 'python', 'javascript', 'html', 'css']):
            return self._suggest_coding_model()
        
        # For creative writing
        if any(keyword in query_lower for keyword in ['write', 'story', 'poem', 'creative', 'blog', 'article']):
            return self._suggest_creative_model()
        
        # For analysis and reasoning
        if any(keyword in query_lower for keyword in ['analyze', 'explain', 'research', 'compare', 'evaluate']):
            return self._suggest_analytical_model()
        
        # For quick questions
        if len(query) < 50 and any(keyword in query_lower for keyword in ['what', 'how', 'when', 'where', 'who']):
            return self._suggest_fast_model()
        
        return None
    
    def _suggest_coding_model(self) -> str:
        """Best models for coding tasks"""
        if 'FIREWORKS_API_KEY' in os.environ:
            return 'accounts/fireworks/models/mixtral-8x7b-instruct'  # Great for code
        elif 'OPENAI_API_KEY' in os.environ:
            return 'gpt-4o'
        else:
            return 'gemini-1.5-pro'
    
    def _suggest_creative_model(self) -> str:
        """Best models for creative tasks"""
        if 'OPENAI_API_KEY' in os.environ:
            return 'gpt-4o'  # Excellent creativity
        elif 'GEMINI_API_KEY' in os.environ:
            return 'gemini-1.5-pro'
        else:
            return 'accounts/fireworks/models/llama-v3p1-70b-instruct'
    
    def _suggest_analytical_model(self) -> str:
        """Best models for analysis"""
        if 'GEMINI_API_KEY' in os.environ:
            return 'gemini-1.5-pro'  # Excellent for analysis
        elif 'OPENAI_API_KEY' in os.environ:
            return 'o1'  # Best reasoning
        else:
            return 'accounts/fireworks/models/llama-v3p1-405b-instruct'
    
    def _suggest_fast_model(self) -> str:
        """Best models for quick responses"""
        if 'GEMINI_API_KEY' in os.environ:
            return 'gemini-1.5-flash'  # Fastest
        elif 'FIREWORKS_API_KEY' in os.environ:
            return 'accounts/fireworks/models/llama-v3p1-8b-instruct'
        else:
            return 'gpt-4o-mini'
    
    def enhance_prompt(self, original_prompt: str, chat_id: int, query: str) -> str:
        """Enhance the system prompt based on context and user preferences"""
        
        enhancements = []
        
        # Add personality based on time of day
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            enhancements.append("You're energetic and ready to tackle the day.")
        elif 12 <= current_hour < 17:
            enhancements.append("You're focused and productive.")
        elif 17 <= current_hour < 22:
            enhancements.append("You're relaxed and conversational.")
        else:
            enhancements.append("You're calm and thoughtful.")
        
        # Add conversation context
        if chat_id in self.conversation_analytics:
            analytics = self.conversation_analytics[chat_id]
            if analytics.get('frequent_topics'):
                top_topic = max(analytics['frequent_topics'].items(), key=lambda x: x[1])[0]
                enhancements.append(f"The user often discusses {top_topic}, so you can reference relevant knowledge.")
        
        # Add query-specific enhancements
        if '?' in query:
            enhancements.append("The user is asking a question, so be direct and informative.")
        if len(query) > 200:
            enhancements.append("This is a detailed query, so provide a comprehensive response.")
        
        # Combine enhancements
        if enhancements:
            enhanced_prompt = f"{original_prompt}\n\nContext: {' '.join(enhancements)}"
            return enhanced_prompt
        
        return original_prompt
    
    def analyze_conversation(self, chat_id: int, query: str, response: str):
        """Analyze conversation patterns for optimization"""
        
        if chat_id not in self.conversation_analytics:
            self.conversation_analytics[chat_id] = {
                'message_count': 0,
                'avg_response_length': 0,
                'frequent_topics': {},
                'preferred_style': 'balanced',
                'last_activity': datetime.now()
            }
        
        analytics = self.conversation_analytics[chat_id]
        analytics['message_count'] += 1
        analytics['last_activity'] = datetime.now()
        
        # Update average response length preference
        current_avg = analytics['avg_response_length']
        response_len = len(response)
        analytics['avg_response_length'] = (current_avg + response_len) / 2
        
        # Extract topics from query
        topics = self._extract_topics(query)
        for topic in topics:
            analytics['frequent_topics'][topic] = analytics['frequent_topics'].get(topic, 0) + 1
        
        # Analyze preferred communication style
        if any(word in query.lower() for word in ['quick', 'brief', 'short']):
            analytics['preferred_style'] = 'concise'
        elif any(word in query.lower() for word in ['detailed', 'explain', 'elaborate']):
            analytics['preferred_style'] = 'detailed'
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        topics = []
        text_lower = text.lower()
        
        # Define topic keywords
        topic_map = {
            'programming': ['code', 'programming', 'python', 'javascript', 'html', 'css', 'api'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'research'],
            'business': ['business', 'marketing', 'sales', 'strategy', 'company'],
            'technology': ['tech', 'ai', 'machine learning', 'software', 'hardware'],
            'writing': ['write', 'writing', 'blog', 'article', 'content', 'story'],
            'education': ['learn', 'study', 'school', 'course', 'tutorial'],
            'health': ['health', 'fitness', 'medical', 'doctor', 'exercise'],
            'travel': ['travel', 'trip', 'vacation', 'flight', 'hotel'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal'],
            'entertainment': ['movie', 'music', 'game', 'book', 'show']
        }
        
        for topic, keywords in topic_map.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def get_cost_optimized_settings(self, query: str, chat_id: int) -> Dict:
        """Optimize settings for cost efficiency"""
        
        # Base optimization settings
        optimized = {
            'max_tokens': self.config.get('MAX_TOKENS', 1200),
            'temperature': self.config.get('TEMPERATURE', 1.0)
        }
        
        # For simple queries, reduce tokens
        if len(query) < 50:
            optimized['max_tokens'] = min(optimized['max_tokens'], 500)
        
        # For factual queries, reduce temperature
        if any(word in query.lower() for word in ['what is', 'define', 'when did', 'who is']):
            optimized['temperature'] = 0.3
        
        # For creative queries, increase temperature
        if any(word in query.lower() for word in ['creative', 'imagine', 'write', 'story']):
            optimized['temperature'] = 1.2
        
        return optimized
    
    def add_smart_formatting(self, response: str, query: str) -> str:
        """Add smart formatting and enhancements to responses"""
        
        # Add emojis for better engagement
        if 'coding' in query.lower() or 'programming' in query.lower():
            response = f"💻 {response}"
        elif any(word in query.lower() for word in ['creative', 'story', 'write']):
            response = f"✨ {response}"
        elif any(word in query.lower() for word in ['analyze', 'research']):
            response = f"🔍 {response}"
        elif any(word in query.lower() for word in ['help', 'support']):
            response = f"🤝 {response}"
        
        # Add helpful tips
        if 'python' in query.lower():
            response += "\n\n💡 *Tip: I can help you debug code, explain concepts, or write scripts!*"
        elif 'write' in query.lower():
            response += "\n\n💡 *Tip: I can help with different writing styles - just ask for formal, casual, or creative!*"
        
        # Add follow-up suggestions
        if len(response) > 500:
            response += "\n\n❓ *Would you like me to elaborate on any part of this?*"
        
        return response
    
    def get_conversation_summary(self, chat_id: int) -> str:
        """Generate a helpful conversation summary"""
        
        if chat_id not in self.conversation_analytics:
            return "This is our first conversation! 👋"
        
        analytics = self.conversation_analytics[chat_id]
        
        summary_parts = []
        summary_parts.append(f"📊 **Conversation Stats:**")
        summary_parts.append(f"• Messages exchanged: {analytics['message_count']}")
        summary_parts.append(f"• Communication style: {analytics['preferred_style']}")
        
        if analytics['frequent_topics']:
            top_topics = sorted(analytics['frequent_topics'].items(), key=lambda x: x[1], reverse=True)[:3]
            topics_str = ", ".join([topic for topic, _ in top_topics])
            summary_parts.append(f"• Main topics: {topics_str}")
        
        last_active = analytics['last_activity']
        if datetime.now() - last_active > timedelta(days=1):
            summary_parts.append(f"• Welcome back! Last active: {last_active.strftime('%Y-%m-%d')}")
        
        return "\n".join(summary_parts)


class ResponseOptimizer:
    """Optimize responses for quality and engagement"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def score_response_quality(self, query: str, response: str) -> float:
        """Score response quality (0-1)"""
        score = 0.0
        
        # Length appropriateness (30% of score)
        query_len = len(query)
        response_len = len(response)
        
        if query_len < 50:  # Short query
            ideal_range = (50, 300)
        elif query_len < 150:  # Medium query
            ideal_range = (200, 800)
        else:  # Long query
            ideal_range = (400, 1500)
        
        if ideal_range[0] <= response_len <= ideal_range[1]:
            score += 0.3
        else:
            # Penalize for being too short or too long
            ratio = response_len / ((ideal_range[0] + ideal_range[1]) / 2)
            score += 0.3 * max(0, 1 - abs(ratio - 1))
        
        # Relevance indicators (40% of score)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Check for keyword overlap
        if query_words & response_words:
            overlap_ratio = len(query_words & response_words) / len(query_words)
            score += 0.2 * overlap_ratio
        
        # Check for direct answers to questions
        if '?' in query:
            if any(starter in response.lower() for starter in ['yes', 'no', 'the answer', 'it is', 'they are']):
                score += 0.2
        
        # Engagement factors (30% of score)
        engagement_indicators = [
            '!' in response,  # Enthusiasm
            '?' in response,  # Follow-up questions
            any(emoji in response for emoji in ['😊', '👍', '💡', '🔍', '✨', '🚀']),  # Emojis
            'tip:' in response.lower(),  # Helpful tips
            'example' in response.lower(),  # Examples provided
        ]
        
        engagement_score = sum(engagement_indicators) / len(engagement_indicators)
        score += 0.3 * engagement_score
        
        return min(1.0, score)
    
    def suggest_improvements(self, query: str, response: str, score: float) -> List[str]:
        """Suggest improvements for low-scoring responses"""
        suggestions = []
        
        if score < 0.3:
            suggestions.append("Consider providing more relevant information")
            suggestions.append("Try to directly address the user's question")
        
        if len(response) < 50:
            suggestions.append("Response might be too brief - consider elaborating")
        elif len(response) > 1000:
            suggestions.append("Response might be too long - consider being more concise")
        
        if '?' in query and '?' not in response:
            suggestions.append("Consider asking follow-up questions to engage the user")
        
        if not any(char in response for char in ['!', '?', '😊', '👍', '💡']):
            suggestions.append("Add some personality with emojis or enthusiasm")
        
        return suggestions


def initialize_enhanced_features(ai_helper, config: dict):
    """Initialize enhanced features for the bot"""
    
    enhanced_bot = EnhancedBot(ai_helper, config)
    response_optimizer = ResponseOptimizer()
    
    # Log initialization
    logging.info("🚀 Enhanced features initialized!")
    logging.info("Features: Smart model suggestions, conversation analysis, cost optimization")
    
    return enhanced_bot, response_optimizer


# Utility functions for advanced features
def detect_language(text: str) -> str:
    """Simple language detection"""
    # Basic language patterns
    if any(word in text.lower() for word in ['the', 'and', 'is', 'are', 'you', 'have']):
        return 'en'
    elif any(word in text.lower() for word in ['el', 'la', 'es', 'son', 'que', 'de']):
        return 'es'
    elif any(word in text.lower() for word in ['le', 'la', 'est', 'sont', 'que', 'de']):
        return 'fr'
    elif any(word in text.lower() for word in ['der', 'die', 'das', 'ist', 'sind', 'und']):
        return 'de'
    else:
        return 'en'  # Default to English


def format_code_response(response: str) -> str:
    """Format code in responses for better readability"""
    
    # Detect code blocks and add syntax highlighting hints
    code_patterns = [
        (r'(def\s+\w+.*?:)', r'```python\n\1\n```'),
        (r'(function\s+\w+.*?\{)', r'```javascript\n\1\n```'),
        (r'(<[^>]+>)', r'```html\n\1\n```'),
    ]
    
    formatted_response = response
    for pattern, replacement in code_patterns:
        formatted_response = re.sub(pattern, replacement, formatted_response)
    
    return formatted_response


def add_helpful_context(response: str, query: str) -> str:
    """Add helpful context and related information"""
    
    context_additions = []
    
    # Programming context
    if any(lang in query.lower() for lang in ['python', 'javascript', 'html', 'css']):
        context_additions.append("\n💡 *Need help with debugging? Just paste your code and describe the issue!*")
    
    # Learning context
    if any(word in query.lower() for word in ['learn', 'tutorial', 'how to']):
        context_additions.append("\n📚 *I can break this down into steps or provide examples if helpful!*")
    
    # Creative context
    if any(word in query.lower() for word in ['write', 'creative', 'story']):
        context_additions.append("\n✨ *I can help with different styles: formal, casual, creative, or technical!*")
    
    return response + "".join(context_additions)