# Quick Start: Use Gemini or Fireworks AI (No OpenAI Required!)

## 🚀 Option 1: Use Google Gemini (Recommended)

### 1. Get Gemini API Key
1. Go to [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Click "Get API Key" 
4. Create a new API key
5. Copy the key

### 2. Set Environment Variables
Create a `.env` file:
```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
AI_MODEL=gemini-1.5-flash
GEMINI_API_KEY=your_gemini_api_key_here

# Optional settings
SHOW_USAGE=true
STREAM=true
BOT_LANGUAGE=en
ADMIN_USER_IDS=-
ALLOWED_TELEGRAM_USER_IDS=*
```

### 3. Run the Bot
```bash
pip install -r requirements.txt
python bot/main.py
```

## 🔥 Option 2: Use Fireworks AI (Cost-Effective)

### 1. Get Fireworks API Key
1. Go to [Fireworks AI](https://fireworks.ai/)
2. Sign up for an account
3. Navigate to your dashboard
4. Generate an API key
5. Copy the key

### 2. Set Environment Variables
Create a `.env` file:
```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
AI_MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct
FIREWORKS_API_KEY=your_fireworks_api_key_here

# Optional settings
SHOW_USAGE=true
STREAM=true
BOT_LANGUAGE=en
ADMIN_USER_IDS=-
ALLOWED_TELEGRAM_USER_IDS=*
```

### 3. Run the Bot
```bash
pip install -r requirements.txt
python bot/main.py
```

## 🎯 Available Models

### Google Gemini Models
- `gemini-2.0-flash-exp` - Latest experimental model
- `gemini-1.5-pro` - Best for complex tasks
- `gemini-1.5-flash` - Fast and efficient (recommended)
- `gemini-pro` - Basic model

### Fireworks AI Models
- `accounts/fireworks/models/llama-v3p1-405b-instruct` - Largest model
- `accounts/fireworks/models/llama-v3p1-70b-instruct` - Balanced performance
- `accounts/fireworks/models/llama-v3p1-8b-instruct` - Fast and cost-effective
- `accounts/fireworks/models/mixtral-8x7b-instruct` - Good for coding
- `accounts/fireworks/models/qwen2p5-72b-instruct` - Multilingual

## ✅ What Works

### With Gemini:
- ✅ Chat conversations
- ✅ Vision (image analysis) 
- ✅ Long context (up to 2M tokens)
- ✅ Multiple languages
- ✅ Streaming responses

### With Fireworks AI:
- ✅ Chat conversations  
- ✅ Function calling (plugins)
- ✅ Fast responses
- ✅ Cost-effective
- ✅ Streaming responses

## ❌ What Doesn't Work (Yet)

### With Gemini:
- ❌ Image generation (use OpenAI for this)
- ❌ Text-to-speech (use OpenAI for this)
- ❌ Audio transcription (use OpenAI for this)

### With Fireworks AI:
- ❌ Vision/image analysis
- ❌ Image generation
- ❌ Text-to-speech  
- ❌ Audio transcription

## 🔧 Troubleshooting

### Error: "OPENAI_API_KEY environment variable is missing"
**Solution**: Make sure you've set `AI_MODEL` to a non-OpenAI model:
```env
AI_MODEL=gemini-1.5-flash  # Not gpt-4o!
GEMINI_API_KEY=your_key
```

### Error: "Unknown model format"
**Solution**: Use exact model names from the lists above.

### Bot not responding
**Solution**: Check your API key is valid and you have credits/quota available.

## 💡 Pro Tips

1. **Start with Gemini 1.5 Flash** - Great balance of speed, quality, and cost
2. **Use Fireworks for high-volume** - Much cheaper for lots of messages  
3. **Keep OpenAI for media** - Add OpenAI key later if you need image generation/TTS
4. **Check your quotas** - Both providers have rate limits

## 🆘 Need Help?

1. Check the [Migration Guide](MIGRATION_GUIDE.md) for detailed information
2. Look at [Issues](https://github.com/n3d1117/chatgpt-telegram-bot/issues) for common problems
3. Make sure your `.env` file is in the root directory (same level as `bot/` folder)

**That's it! No OpenAI required! 🎉**