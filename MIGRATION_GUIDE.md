# Migration Guide: Multi-Provider AI Support

This guide helps you migrate from the OpenAI-only version to the new multi-provider version that supports OpenAI, Google Gemini, and Fireworks AI.

## What's New

### 🚀 Multi-Provider Support
- **OpenAI**: All existing GPT models (GPT-4o, GPT-4, GPT-3.5-turbo, o1, o1-mini)
- **Google Gemini**: Latest Gemini models (gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash, gemini-pro)
- **Fireworks AI**: Open-source models (Llama 3.1, Mixtral, Qwen2.5)

### 🔄 Seamless Model Switching
Change AI providers by simply updating one environment variable - no code changes needed!

## Migration Steps

### 1. Update Dependencies

The new requirements include additional packages for multi-provider support:

```bash
pip install google-generativeai>=0.8.0 fireworks-ai>=0.15.0
```

Or use the updated `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Update Environment Variables

#### Old Configuration (OpenAI only)
```env
OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=your_openai_key
```

#### New Configuration (Multi-provider)
```env
# Choose any supported model
AI_MODEL=gpt-4o                    # OpenAI model
# AI_MODEL=gemini-1.5-pro          # Or Gemini model  
# AI_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct  # Or Fireworks model

# Add only the API keys you need
OPENAI_API_KEY=your_openai_key      # For OpenAI models
GEMINI_API_KEY=your_gemini_key      # For Gemini models
FIREWORKS_API_KEY=your_fireworks_key # For Fireworks models
```

### 3. Backward Compatibility

Don't worry! Your existing configuration still works:
- `OPENAI_MODEL` is still supported (automatically mapped to `AI_MODEL`)
- `OPENAI_API_KEY` continues to work for OpenAI models
- All existing environment variables remain functional

## Supported Models & Features

| Provider | Model | Chat | Vision | Functions | Image Gen | TTS | Transcription |
|----------|-------|------|--------|-----------|-----------|-----|---------------|
| **OpenAI** | gpt-4o | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **OpenAI** | gpt-4o-mini | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **OpenAI** | gpt-4 | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **OpenAI** | gpt-3.5-turbo | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **OpenAI** | o1 / o1-mini | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Gemini** | gemini-2.0-flash-exp | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| **Gemini** | gemini-1.5-pro | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| **Gemini** | gemini-1.5-flash | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| **Gemini** | gemini-pro | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Fireworks** | llama-v3p1-405b | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| **Fireworks** | llama-v3p1-70b | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| **Fireworks** | llama-v3p1-8b | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| **Fireworks** | mixtral-8x7b | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |

**Legend:**
- ✅ Fully supported
- ⚠️ Planned/limited support  
- ❌ Not available

## Getting API Keys

### OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Sign up or log in
3. Create a new API key
4. Add billing information for paid features

### Google Gemini
1. Visit [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your environment

### Fireworks AI
1. Visit [Fireworks AI](https://fireworks.ai/)
2. Sign up for an account
3. Navigate to API settings
4. Generate an API key

## Example Configurations

### Use GPT-4o (OpenAI)
```env
AI_MODEL=gpt-4o
OPENAI_API_KEY=your_openai_key
TELEGRAM_BOT_TOKEN=your_telegram_token
```

### Use Gemini 1.5 Pro (Google)
```env
AI_MODEL=gemini-1.5-pro
GEMINI_API_KEY=your_gemini_key
TELEGRAM_BOT_TOKEN=your_telegram_token
```

### Use Llama 3.1 70B (Fireworks)
```env
AI_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct
FIREWORKS_API_KEY=your_fireworks_key
TELEGRAM_BOT_TOKEN=your_telegram_token
```

## Troubleshooting

### Common Issues

**1. Missing API Key Error**
```
Error: GEMINI_API_KEY environment variable is missing
```
**Solution**: Add the required API key for your chosen model's provider.

**2. Model Not Found**
```
Error: Unknown model: my-custom-model
```
**Solution**: Use one of the supported models listed in the documentation.

**3. Function Calling Not Working**
```
Warning: ENABLE_FUNCTIONS is set to true, but the model does not support it
```
**Solution**: Disable functions or switch to a model that supports them (most models except o1 series).

### Performance Considerations

- **OpenAI**: Generally fastest response times, highest cost
- **Gemini**: Very fast, good for high-volume usage, excellent context length
- **Fireworks**: Cost-effective for open-source models, good performance

### Rate Limits

Each provider has different rate limits:
- **OpenAI**: Depends on your tier (free/paid)
- **Gemini**: Generous free tier, then paid usage
- **Fireworks**: Varies by model and subscription

## Advanced Usage

### Switching Models Dynamically
You can change models without restarting by updating the environment variable and restarting the bot:

```bash
# Update your .env file
echo "AI_MODEL=gemini-1.5-pro" >> .env

# Restart the bot
docker-compose restart  # or however you run the bot
```

### Cost Optimization
- Use **Fireworks** for general chat (lowest cost)
- Use **Gemini** for long conversations (best context length)
- Use **OpenAI** for advanced features (vision, image generation, TTS)

## Support

If you encounter any issues during migration:

1. Check the [Issues](https://github.com/n3d1117/chatgpt-telegram-bot/issues) section
2. Ensure all required API keys are set
3. Verify your model name is exactly as listed in the documentation
4. Check the logs for specific error messages

## Contributing

Help us improve multi-provider support:
- Test with different models and report issues
- Contribute provider-specific optimizations
- Help with documentation and examples

Happy chatting with multiple AI providers! 🤖✨