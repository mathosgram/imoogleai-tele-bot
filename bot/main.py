import logging
import os

from dotenv import load_dotenv

from plugin_manager import PluginManager
from ai_helper import AIHelper, default_max_tokens, are_functions_available
from telegram_bot import ChatGPTTelegramBot


def main():
    # Read .env file
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Check if the required environment variables are set
    required_values = ['TELEGRAM_BOT_TOKEN']
    
    # Determine which AI provider API keys are needed based on the model
cursor/configure-ai-models-to-use-gemini-and-fireworks-da07
    model = os.environ.get('AI_MODEL', os.environ.get('OPENAI_MODEL', None))
    
    # Only require API keys if a model is specified
    if model:
        # Check for provider-specific API keys
        if model.startswith('gpt-') or model.startswith('o1') or model.startswith('davinci') or model.startswith('text-'):
            required_values.append('OPENAI_API_KEY')
        elif model.startswith('gemini'):
            required_values.append('GEMINI_API_KEY')
        elif 'fireworks' in model or model.startswith('accounts/fireworks'):
            required_values.append('FIREWORKS_API_KEY')
        else:
            # If model format is unclear, check which API keys are available
            if os.environ.get('OPENAI_API_KEY'):
                logging.info(f"Unknown model format '{model}', but OPENAI_API_KEY is available, assuming OpenAI model")
            elif os.environ.get('GEMINI_API_KEY'):
                logging.info(f"Unknown model format '{model}', but GEMINI_API_KEY is available, assuming Gemini model")
            elif os.environ.get('FIREWORKS_API_KEY'):
                logging.info(f"Unknown model format '{model}', but FIREWORKS_API_KEY is available, assuming Fireworks model")
            else:
                logging.error(f"Unknown model format '{model}' and no API keys found. Please set AI_MODEL to a supported model and provide the corresponding API key.")
                exit(1)
    else:
        # No model specified, check which API keys are available and set default
        if os.environ.get('OPENAI_API_KEY'):
            model = 'gpt-4o'
            logging.info("No AI_MODEL specified, defaulting to gpt-4o (OpenAI)")
        elif os.environ.get('GEMINI_API_KEY'):
            model = 'gemini-1.5-flash'
            logging.info("No AI_MODEL specified, defaulting to gemini-1.5-flash (Google)")
        elif os.environ.get('FIREWORKS_API_KEY'):
            model = 'accounts/fireworks/models/llama-v3p1-8b-instruct'
            logging.info("No AI_MODEL specified, defaulting to llama-v3p1-8b-instruct (Fireworks)")
        else:
            logging.error("No AI_MODEL specified and no API keys found. Please set AI_MODEL and provide the corresponding API key.")
            logging.error("Example: AI_MODEL=gemini-1.5-flash and GEMINI_API_KEY=your_key")
            exit(1)

    model = os.environ.get('AI_MODEL', 'gpt-4o')
    
    # Check for provider-specific API keys
    if model.startswith('gpt-') or model.startswith('o1') or model.startswith('davinci') or model.startswith('text-'):
        required_values.append('OPENAI_API_KEY')
    elif model.startswith('gemini'):
        required_values.append('GEMINI_API_KEY')
    elif 'fireworks' in model or model.startswith('accounts/fireworks'):
        required_values.append('FIREWORKS_API_KEY')
    else:
        # Default to OpenAI if model format is unclear
        required_values.append('OPENAI_API_KEY')

    
    missing_values = [value for value in required_values if os.environ.get(value) is None]
    if len(missing_values) > 0:
        logging.error(f'The following environment values are missing in your .env: {", ".join(missing_values)}')
        exit(1)

    # Setup configurations
     cursor/configure-ai-models-to-use-gemini-and-fireworks-da07
    # Model is already determined from the validation logic above

    # Use AI_MODEL instead of OPENAI_MODEL for generic model selection
    model = os.environ.get('AI_MODEL', os.environ.get('OPENAI_MODEL', 'gpt-4o'))

    functions_available = are_functions_available(model=model)
    max_tokens_default = default_max_tokens(model=model)
    
    ai_config = {
        # Provider API keys
        'openai_api_key': os.environ.get('OPENAI_API_KEY'),
        'gemini_api_key': os.environ.get('GEMINI_API_KEY'),
        'fireworks_api_key': os.environ.get('FIREWORKS_API_KEY'),
        
        # General configuration
        'show_usage': os.environ.get('SHOW_USAGE', 'false').lower() == 'true',
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('OPENAI_PROXY', None),
        'max_history_size': int(os.environ.get('MAX_HISTORY_SIZE', 15)),
        'max_conversation_age_minutes': int(os.environ.get('MAX_CONVERSATION_AGE_MINUTES', 180)),
        'assistant_prompt': os.environ.get('ASSISTANT_PROMPT', 'You are a helpful assistant.'),
        'max_tokens': int(os.environ.get('MAX_TOKENS', max_tokens_default)),
        'n_choices': int(os.environ.get('N_CHOICES', 1)),
        'temperature': float(os.environ.get('TEMPERATURE', 1.0)),
        
        # Model configuration
        'model': model,
        'enable_functions': os.environ.get('ENABLE_FUNCTIONS', str(functions_available)).lower() == 'true',
        'functions_max_consecutive_calls': int(os.environ.get('FUNCTIONS_MAX_CONSECUTIVE_CALLS', 10)),
        'presence_penalty': float(os.environ.get('PRESENCE_PENALTY', 0.0)),
        'frequency_penalty': float(os.environ.get('FREQUENCY_PENALTY', 0.0)),
        'bot_language': os.environ.get('BOT_LANGUAGE', 'en'),
        'show_plugins_used': os.environ.get('SHOW_PLUGINS_USED', 'false').lower() == 'true',
        
        # Image generation (mainly for OpenAI)
        'image_model': os.environ.get('IMAGE_MODEL', 'dall-e-2'),
        'image_quality': os.environ.get('IMAGE_QUALITY', 'standard'),
        'image_style': os.environ.get('IMAGE_STYLE', 'vivid'),
        'image_size': os.environ.get('IMAGE_SIZE', '512x512'),
        
        # Audio/Speech configuration
        'whisper_prompt': os.environ.get('WHISPER_PROMPT', ''),
        'tts_model': os.environ.get('TTS_MODEL', 'tts-1'),
        'tts_voice': os.environ.get('TTS_VOICE', 'alloy'),
        
        # Vision configuration
        'vision_model': os.environ.get('VISION_MODEL', model),  # Use the main model for vision by default
        'enable_vision_follow_up_questions': os.environ.get('ENABLE_VISION_FOLLOW_UP_QUESTIONS', 'true').lower() == 'true',
        'vision_prompt': os.environ.get('VISION_PROMPT', 'What is in this image'),
        'vision_detail': os.environ.get('VISION_DETAIL', 'auto'),
        'vision_max_tokens': int(os.environ.get('VISION_MAX_TOKENS', '300')),
    }

    if ai_config['enable_functions'] and not functions_available:
        logging.error(f'ENABLE_FUNCTIONS is set to true, but the model {model} does not support it. '
                        'Please set ENABLE_FUNCTIONS to false or use a model that supports it.')
        exit(1)
    if os.environ.get('MONTHLY_USER_BUDGETS') is not None:
        logging.warning('The environment variable MONTHLY_USER_BUDGETS is deprecated. '
                        'Please use USER_BUDGETS with BUDGET_PERIOD instead.')
    if os.environ.get('MONTHLY_GUEST_BUDGET') is not None:
        logging.warning('The environment variable MONTHLY_GUEST_BUDGET is deprecated. '
                        'Please use GUEST_BUDGET with BUDGET_PERIOD instead.')

    telegram_config = {
        'token': os.environ['TELEGRAM_BOT_TOKEN'],
        'admin_user_ids': os.environ.get('ADMIN_USER_IDS', '-'),
        'allowed_user_ids': os.environ.get('ALLOWED_TELEGRAM_USER_IDS', '*'),
        'enable_quoting': os.environ.get('ENABLE_QUOTING', 'true').lower() == 'true',
        'enable_image_generation': os.environ.get('ENABLE_IMAGE_GENERATION', 'true').lower() == 'true',
        'enable_transcription': os.environ.get('ENABLE_TRANSCRIPTION', 'true').lower() == 'true',
        'enable_vision': os.environ.get('ENABLE_VISION', 'true').lower() == 'true',
        'enable_tts_generation': os.environ.get('ENABLE_TTS_GENERATION', 'true').lower() == 'true',
        'budget_period': os.environ.get('BUDGET_PERIOD', 'monthly').lower(),
        'user_budgets': os.environ.get('USER_BUDGETS', os.environ.get('MONTHLY_USER_BUDGETS', '*')),
        'guest_budget': float(os.environ.get('GUEST_BUDGET', os.environ.get('MONTHLY_GUEST_BUDGET', '100.0'))),
        'stream': os.environ.get('STREAM', 'true').lower() == 'true',
        'proxy': os.environ.get('PROXY', None) or os.environ.get('TELEGRAM_PROXY', None),
        'voice_reply_transcript': os.environ.get('VOICE_REPLY_WITH_TRANSCRIPT_ONLY', 'false').lower() == 'true',
        'voice_reply_prompts': os.environ.get('VOICE_REPLY_PROMPTS', '').split(';'),
        'ignore_group_transcriptions': os.environ.get('IGNORE_GROUP_TRANSCRIPTIONS', 'true').lower() == 'true',
        'ignore_group_vision': os.environ.get('IGNORE_GROUP_VISION', 'true').lower() == 'true',
        'group_trigger_keyword': os.environ.get('GROUP_TRIGGER_KEYWORD', ''),
        'token_price': float(os.environ.get('TOKEN_PRICE', 0.002)),
        'image_prices': [float(i) for i in os.environ.get('IMAGE_PRICES', "0.016,0.018,0.02").split(",")],
        'vision_token_price': float(os.environ.get('VISION_TOKEN_PRICE', '0.01')),
        'image_receive_mode': os.environ.get('IMAGE_FORMAT', "photo"),
        'tts_model': os.environ.get('TTS_MODEL', 'tts-1'),
        'tts_prices': [float(i) for i in os.environ.get('TTS_PRICES', "0.015,0.030").split(",")],
        'transcription_price': float(os.environ.get('TRANSCRIPTION_PRICE', 0.006)),
        'bot_language': os.environ.get('BOT_LANGUAGE', 'en'),
    }

    plugin_config = {
        'plugins': os.environ.get('PLUGINS', '').split(',')
    }

    # Setup and run AI-powered Telegram bot
    plugin_manager = PluginManager(config=plugin_config)
    ai_helper = AIHelper(config=ai_config, plugin_manager=plugin_manager)
    telegram_bot = ChatGPTTelegramBot(config=telegram_config, openai=ai_helper)
    telegram_bot.run()


if __name__ == '__main__':
    main()
