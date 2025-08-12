from __future__ import annotations
import datetime
import logging
import os
import abc
from typing import Dict, List, Tuple, Optional, Union, AsyncGenerator

import tiktoken
import json
import httpx
import io
from PIL import Image

# Import all provider libraries
import openai
import google.generativeai as genai
import fireworks.client

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from utils import is_direct_result, encode_image, decode_image
from plugin_manager import PluginManager


# Model definitions for different providers
OPENAI_MODELS = {
    "gpt-3.5-turbo": {"context": 4096, "supports_functions": True, "supports_vision": False},
    "gpt-4": {"context": 8192, "supports_functions": True, "supports_vision": False},
    "gpt-4o": {"context": 128000, "supports_functions": True, "supports_vision": True},
    "gpt-4o-mini": {"context": 128000, "supports_functions": True, "supports_vision": True},
    "o1": {"context": 100000, "supports_functions": False, "supports_vision": False},
    "o1-mini": {"context": 65536, "supports_functions": False, "supports_vision": False},
}

GEMINI_MODELS = {
    "gemini-2.0-flash-exp": {"context": 1000000, "supports_functions": True, "supports_vision": True},
    "gemini-1.5-pro": {"context": 2000000, "supports_functions": True, "supports_vision": True},
    "gemini-1.5-flash": {"context": 1000000, "supports_functions": True, "supports_vision": True},
    "gemini-pro": {"context": 32000, "supports_functions": False, "supports_vision": False},
}

FIREWORKS_MODELS = {
    "accounts/fireworks/models/llama-v3p1-405b-instruct": {"context": 131072, "supports_functions": True, "supports_vision": False},
    "accounts/fireworks/models/llama-v3p1-70b-instruct": {"context": 131072, "supports_functions": True, "supports_vision": False},
    "accounts/fireworks/models/llama-v3p1-8b-instruct": {"context": 131072, "supports_functions": True, "supports_vision": False},
    "accounts/fireworks/models/mixtral-8x7b-instruct": {"context": 32768, "supports_functions": True, "supports_vision": False},
    "accounts/fireworks/models/qwen2p5-72b-instruct": {"context": 131072, "supports_functions": True, "supports_vision": False},
}

ALL_MODELS = {**OPENAI_MODELS, **GEMINI_MODELS, **FIREWORKS_MODELS}


def default_max_tokens(model: str) -> int:
    """
    Gets the default number of max tokens for the given model.
    :param model: The model name
    :return: The default number of max tokens
    """
    if model in ALL_MODELS:
        context_length = ALL_MODELS[model]["context"]
        # Reserve 20% for output, 80% for input
        return min(4096, int(context_length * 0.2))
    return 1200  # fallback


def are_functions_available(model: str) -> bool:
    """
    Whether the given model supports functions
    """
    if model in ALL_MODELS:
        return ALL_MODELS[model]["supports_functions"]
    return False


def supports_vision(model: str) -> bool:
    """
    Whether the given model supports vision
    """
    if model in ALL_MODELS:
        return ALL_MODELS[model]["supports_vision"]
    return False


def get_provider_from_model(model: str) -> str:
    """
    Determine the provider based on the model name
    """
    if model in OPENAI_MODELS:
        return "openai"
    elif model in GEMINI_MODELS:
        return "gemini"
    elif model in FIREWORKS_MODELS:
        return "fireworks"
    else:
        raise ValueError(f"Unknown model: {model}")


# Load translations
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)
translations_file_path = os.path.join(parent_dir_path, 'translations.json')
with open(translations_file_path, 'r', encoding='utf-8') as f:
    translations = json.load(f)


def localized_text(key, bot_language):
    """
    Return translated text for a key in specified bot_language.
    Keys and translations can be found in the translations.json.
    """
    try:
        return translations[bot_language][key]
    except KeyError:
        logging.warning(f"No translation available for bot_language code '{bot_language}' and key '{key}'")
        # Fallback to English if the translation is not available
        if key in translations['en']:
            return translations['en'][key]
        else:
            logging.warning(f"No english definition found for key '{key}' in translations.json")
            return key


class BaseAIProvider(abc.ABC):
    """
    Abstract base class for AI providers
    """
    
    def __init__(self, config: dict):
        self.config = config
    
    @abc.abstractmethod
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Union[object, AsyncGenerator]:
        """Generate chat completion"""
        pass
    
    @abc.abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> str:
        """Generate image from prompt"""
        pass
    
    @abc.abstractmethod
    async def transcribe_audio(self, audio_file, **kwargs) -> str:
        """Transcribe audio to text"""
        pass
    
    @abc.abstractmethod
    async def text_to_speech(self, text: str, **kwargs) -> bytes:
        """Convert text to speech"""
        pass


class OpenAIProvider(BaseAIProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        http_client = httpx.AsyncClient(proxy=config.get('proxy')) if config.get('proxy') else None
        self.client = openai.AsyncOpenAI(api_key=config['openai_api_key'], http_client=http_client)
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Union[object, AsyncGenerator]:
        return await self.client.chat.completions.create(
            model=self.config['model'],
            messages=messages,
            **kwargs
        )
    
    async def generate_image(self, prompt: str, **kwargs) -> str:
        response = await self.client.images.generate(
            model=self.config.get('image_model', 'dall-e-2'),
            prompt=prompt,
            quality=self.config.get('image_quality', 'standard'),
            style=self.config.get('image_style', 'vivid'),
            size=self.config.get('image_size', '512x512'),
            n=1
        )
        return response.data[0].url
    
    async def transcribe_audio(self, audio_file, **kwargs) -> str:
        transcript = await self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt=self.config.get('whisper_prompt', '')
        )
        return transcript.text
    
    async def text_to_speech(self, text: str, **kwargs) -> bytes:
        response = await self.client.audio.speech.create(
            model=self.config.get('tts_model', 'tts-1'),
            voice=self.config.get('tts_voice', 'alloy'),
            input=text
        )
        return response.content


class GeminiProvider(BaseAIProvider):
    """Google Gemini provider implementation"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        genai.configure(api_key=config['gemini_api_key'])
        self.client = genai.GenerativeModel(config['model'])
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Union[object, AsyncGenerator]:
        # Convert OpenAI format to Gemini format
        gemini_content = self._convert_messages_to_gemini(messages)
        
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get('temperature', 1.0),
            max_output_tokens=kwargs.get('max_tokens', 1024),
        )
        
        # For Gemini, we need to handle chat format differently
        if len(gemini_content) == 1:
            # Single message
            content = gemini_content[0]
        else:
            # Multi-turn conversation - use chat
            # Separate system message if present
            system_prompt = None
            chat_messages = []
            
            for msg in gemini_content:
                if msg.get('role') == 'system':
                    system_prompt = msg['parts'][0] if isinstance(msg['parts'], list) else msg['parts']
                else:
                    chat_messages.append(msg)
            
            if system_prompt:
                # Include system prompt in the first user message
                if chat_messages and chat_messages[0].get('role') == 'user':
                    if isinstance(chat_messages[0]['parts'], list):
                        chat_messages[0]['parts'].insert(0, f"System: {system_prompt}\n\nUser: ")
                    else:
                        chat_messages[0]['parts'] = f"System: {system_prompt}\n\nUser: {chat_messages[0]['parts']}"
            
            # Use the last message as content for generation
            content = chat_messages[-1]['parts'] if chat_messages else gemini_content[0]['parts']
        
        if kwargs.get('stream', False):
            response = self.client.generate_content(
                content,
                generation_config=generation_config,
                stream=True
            )
            return response
        else:
            response = self.client.generate_content(
                content,
                generation_config=generation_config
            )
            return response
    
    def _convert_messages_to_gemini(self, messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI message format to Gemini format"""
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            
            if isinstance(content, list):
                # Handle multimodal content
                parts = []
                for item in content:
                    if item["type"] == "text":
                        parts.append(item["text"])
                    elif item["type"] == "image_url":
                        # Convert base64 image to Gemini format
                        image_data = decode_image(item["image_url"]["url"])
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        })
                gemini_messages.append({"role": role, "parts": parts})
            else:
                gemini_messages.append({"role": role, "parts": [content]})
        
        return gemini_messages
    
    async def generate_image(self, prompt: str, **kwargs) -> str:
        # Gemini doesn't have image generation, fallback to a placeholder
        raise NotImplementedError("Gemini doesn't support image generation. Consider using OpenAI or another provider.")
    
    async def transcribe_audio(self, audio_file, **kwargs) -> str:
        # Gemini doesn't have dedicated transcription, could use multimodal
        raise NotImplementedError("Gemini audio transcription not implemented yet.")
    
    async def text_to_speech(self, text: str, **kwargs) -> bytes:
        # Gemini doesn't have TTS
        raise NotImplementedError("Gemini doesn't support text-to-speech.")


class FireworksProvider(BaseAIProvider):
    """Fireworks AI provider implementation"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = fireworks.client.Fireworks(api_key=config['fireworks_api_key'])
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Union[object, AsyncGenerator]:
        response = await self.client.chat.completions.acreate(
            model=self.config['model'],
            messages=messages,
            **kwargs
        )
        return response
    
    async def generate_image(self, prompt: str, **kwargs) -> str:
        # Fireworks may have image models, but implementation depends on their specific API
        raise NotImplementedError("Fireworks image generation not implemented yet.")
    
    async def transcribe_audio(self, audio_file, **kwargs) -> str:
        # Fireworks may have audio models
        raise NotImplementedError("Fireworks audio transcription not implemented yet.")
    
    async def text_to_speech(self, text: str, **kwargs) -> bytes:
        # Fireworks may have TTS models
        raise NotImplementedError("Fireworks text-to-speech not implemented yet.")


class AIHelper:
    """
    Multi-provider AI helper class that supports OpenAI, Gemini, and Fireworks AI.
    """

    def __init__(self, config: dict, plugin_manager: PluginManager):
        """
        Initializes the AI helper class with the given configuration.
        :param config: A dictionary containing the AI configuration
        :param plugin_manager: The plugin manager
        """
        self.config = config
        self.plugin_manager = plugin_manager
        self.conversations: dict[int: list] = {}  # {chat_id: history}
        self.conversations_vision: dict[int: bool] = {}  # {chat_id: is_vision}
        self.last_updated: dict[int: datetime] = {}  # {chat_id: last_update_timestamp}
        
        # Initialize the appropriate provider based on the model
        self.provider_name = get_provider_from_model(config['model'])
        
        if self.provider_name == "openai":
            self.provider = OpenAIProvider(config)
        elif self.provider_name == "gemini":
            self.provider = GeminiProvider(config)
        elif self.provider_name == "fireworks":
            self.provider = FireworksProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

    def get_conversation_stats(self, chat_id: int) -> tuple[int, int]:
        """
        Gets the number of messages and tokens used in the conversation.
        :param chat_id: The chat ID
        :return: A tuple containing the number of messages and tokens used
        """
        if chat_id not in self.conversations:
            self.reset_chat_history(chat_id)
        return len(self.conversations[chat_id]), self.__count_tokens(self.conversations[chat_id])

    async def get_chat_response(self, chat_id: int, query: str) -> tuple[str, str]:
        """
        Gets a full response from the AI model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query)
        
        if self.config['enable_functions'] and not self.conversations_vision[chat_id] and self.provider_name == "openai":
            response, plugins_used = await self.__handle_function_call(chat_id, response)
            if is_direct_result(response):
                return response, '0'

        answer = self._extract_response_content(response)
        self.__add_to_history(chat_id, role="assistant", content=answer)

        # Add usage and plugin information
        answer = self._add_response_metadata(answer, response, plugins_used)
        
        tokens_used = self._get_tokens_used(response)
        return answer, str(tokens_used)

    async def get_chat_response_stream(self, chat_id: int, query: str):
        """
        Stream response from the AI model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model and the number of tokens used, or 'not_finished'
        """
        plugins_used = ()
        response = await self.__common_get_chat_response(chat_id, query, stream=True)
        
        if self.config['enable_functions'] and not self.conversations_vision[chat_id] and self.provider_name == "openai":
            response, plugins_used = await self.__handle_function_call(chat_id, response, stream=True)
            if is_direct_result(response):
                yield response, '0'
                return

        answer = ''
        
        if self.provider_name == "openai":
            async for chunk in response:
                if len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    answer += delta.content
                    yield answer, 'not_finished'
        elif self.provider_name == "gemini":
            async for chunk in response:
                if chunk.text:
                    answer += chunk.text
                    yield answer, 'not_finished'
        elif self.provider_name == "fireworks":
            async for chunk in response:
                if len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    answer += delta.content
                    yield answer, 'not_finished'
        
        answer = answer.strip()
        self.__add_to_history(chat_id, role="assistant", content=answer)
        tokens_used = str(self.__count_tokens(self.conversations[chat_id]))

        # Add metadata
        answer = self._add_response_metadata(answer, None, plugins_used, tokens_used)
        yield answer, tokens_used

    def _extract_response_content(self, response) -> str:
        """Extract text content from provider response"""
        if self.provider_name == "openai" or self.provider_name == "fireworks":
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
        elif self.provider_name == "gemini":
            if hasattr(response, 'text'):
                return response.text.strip()
        return ""

    def _get_tokens_used(self, response) -> int:
        """Get token usage from provider response"""
        if self.provider_name == "openai" or self.provider_name == "fireworks":
            if hasattr(response, 'usage'):
                return response.usage.total_tokens
        elif self.provider_name == "gemini":
            if hasattr(response, 'usage_metadata'):
                return response.usage_metadata.total_token_count
        return 0

    def _add_response_metadata(self, answer: str, response, plugins_used, tokens_used=None) -> str:
        """Add usage and plugin information to response"""
        bot_language = self.config['bot_language']
        show_plugins_used = len(plugins_used) > 0 and self.config['show_plugins_used']
        plugin_names = tuple(self.plugin_manager.get_plugin_source_name(plugin) for plugin in plugins_used)
        
        if tokens_used is None:
            tokens_used = self._get_tokens_used(response)
        
        if self.config['show_usage']:
            answer += f"\n\n---\n💰 {str(tokens_used)} {localized_text('stats_tokens', bot_language)}"
            if show_plugins_used:
                answer += f"\n🔌 {', '.join(plugin_names)}"
        elif show_plugins_used:
            answer += f"\n\n---\n🔌 {', '.join(plugin_names)}"
        
        return answer

    async def __common_get_chat_response(self, chat_id: int, query: str, stream=False):
        """
        Request a response from the AI model.
        :param chat_id: The chat ID
        :param query: The query to send to the model
        :return: The answer from the model
        """
        try:
            if chat_id not in self.conversations or self.__max_age_reached(chat_id):
                self.reset_chat_history(chat_id)

            self.last_updated[chat_id] = datetime.datetime.now()
            self.__add_to_history(chat_id, role="user", content=query)

            # Summarize the chat history if it's too long to avoid excessive token usage
            token_count = self.__count_tokens(self.conversations[chat_id])
            exceeded_max_tokens = token_count + self.config['max_tokens'] > self.__max_model_tokens()
            exceeded_max_history_size = len(self.conversations[chat_id]) > self.config['max_history_size']

            if exceeded_max_tokens or exceeded_max_history_size:
                logging.info(f'Chat history for chat ID {chat_id} is too long. Summarising...')
                try:
                    summary = await self.__summarise(self.conversations[chat_id][:-1])
                    logging.debug(f'Summary: {summary}')
                    self.reset_chat_history(chat_id, self.conversations[chat_id][0]['content'])
                    self.__add_to_history(chat_id, role="assistant", content=summary)
                    self.__add_to_history(chat_id, role="user", content=query)
                except Exception as e:
                    logging.warning(f'Error while summarising chat history: {str(e)}. Popping elements instead...')
                    self.conversations[chat_id] = self.conversations[chat_id][-self.config['max_history_size']:]

            # Prepare common arguments
            common_args = {
                'temperature': self.config['temperature'],
                'max_tokens': self.config['max_tokens'],
                'stream': stream
            }

            # Add provider-specific arguments
            if self.provider_name == "openai" or self.provider_name == "fireworks":
                common_args.update({
                    'n': self.config['n_choices'],
                    'presence_penalty': self.config['presence_penalty'],
                    'frequency_penalty': self.config['frequency_penalty'],
                })
                
                # Add function calling for OpenAI if enabled
                if self.config['enable_functions'] and self.provider_name == "openai":
                    functions = self.plugin_manager.get_functions_specs()
                    if len(functions) > 0:
                        common_args['tools'] = [{"type": "function", "function": func} for func in functions]
                        common_args['tool_choice'] = 'auto'

            return await self.provider.chat_completion(self.conversations[chat_id], **common_args)

        except Exception as e:
            bot_language = self.config['bot_language']
            if "rate limit" in str(e).lower():
                raise Exception(f"⚠️ _{localized_text('openai_invalid', bot_language)}._ ⚠️\n{str(e)}") from e
            else:
                raise Exception(f"⚠️ _{localized_text('error', bot_language)}._ ⚠️\n{str(e)}") from e

    # Continue with rest of the helper methods...
    def reset_chat_history(self, chat_id, content=''):
        """
        Resets the conversation history.
        """
        if content == '':
            content = self.config['assistant_prompt']
        self.conversations[chat_id] = [{"role": "system", "content": content}]
        self.conversations_vision[chat_id] = False

    def __max_age_reached(self, chat_id) -> bool:
        """
        Checks if the maximum conversation age has been reached.
        :param chat_id: The chat ID
        :return: A boolean indicating whether the maximum conversation age has been reached
        """
        if chat_id not in self.last_updated:
            return False
        last_updated = self.last_updated[chat_id]
        now = datetime.datetime.now()
        max_age_minutes = self.config['max_conversation_age_minutes']
        return last_updated < now - datetime.timedelta(minutes=max_age_minutes)

    def __add_to_history(self, chat_id: int, role: str, content: str):
        """
        Adds a message to the conversation history.
        :param chat_id: The chat ID
        :param role: The role of the message sender
        :param content: The message content
        """
        self.conversations[chat_id].append({"role": role, "content": content})

    async def __summarise(self, conversation) -> str:
        """
        Summarises the conversation history.
        :param conversation: The conversation history
        :return: The summary
        """
        messages = [
            {"role": "assistant", "content": "Summarize this conversation in 700 characters or less"},
            {"role": "user", "content": str(conversation)}
        ]
        
        response = await self.provider.chat_completion(messages, temperature=0.4, max_tokens=200)
        return self._extract_response_content(response)

    def __max_model_tokens(self):
        """Get maximum context length for current model"""
        if self.config['model'] in ALL_MODELS:
            return ALL_MODELS[self.config['model']]["context"]
        return 4096  # fallback

    def __count_tokens(self, messages) -> int:
        """
        Counts the number of tokens required to send the given messages.
        :param messages: the messages to send
        :return: the number of tokens required
        """
        # This is a simplified token counting - for accurate counting,
        # each provider would need its own implementation
        model = self.config['model']
        try:
            encoding = tiktoken.encoding_for_model(model) if self.provider_name == "openai" else tiktoken.get_encoding("cl100k_base")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += 3  # message overhead
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and 'text' in item:
                            num_tokens += len(encoding.encode(item['text']))
                        elif isinstance(item, str):
                            num_tokens += len(encoding.encode(item))
        num_tokens += 3  # reply overhead
        return num_tokens

    async def __handle_function_call(self, chat_id, response, stream=False, times=0, plugins_used=()):
        """Handle function calling for OpenAI provider"""
        function_name = ''
        arguments = ''
        if stream:
            async for item in response:
                if len(item.choices) > 0:
                    first_choice = item.choices[0]
                    if first_choice.delta and first_choice.delta.tool_calls:
                        # Handle new tools format
                        for tool_call in first_choice.delta.tool_calls:
                            if tool_call.function:
                                if tool_call.function.name:
                                    function_name += tool_call.function.name
                                if tool_call.function.arguments:
                                    arguments += tool_call.function.arguments
                    elif first_choice.delta and first_choice.delta.function_call:
                        # Handle legacy function_call format
                        if first_choice.delta.function_call.name:
                            function_name += first_choice.delta.function_call.name
                        if first_choice.delta.function_call.arguments:
                            arguments += first_choice.delta.function_call.arguments
                    elif first_choice.finish_reason and first_choice.finish_reason in ['function_call', 'tool_calls']:
                        break
                    else:
                        return response, plugins_used
                else:
                    return response, plugins_used
        else:
            if len(response.choices) > 0:
                first_choice = response.choices[0]
                if first_choice.message.tool_calls:
                    # Handle new tools format
                    tool_call = first_choice.message.tool_calls[0]
                    if tool_call.function:
                        function_name = tool_call.function.name
                        arguments = tool_call.function.arguments
                elif first_choice.message.function_call:
                    # Handle legacy function_call format
                    if first_choice.message.function_call.name:
                        function_name += first_choice.message.function_call.name
                    if first_choice.message.function_call.arguments:
                        arguments += first_choice.message.function_call.arguments
                else:
                    return response, plugins_used
            else:
                return response, plugins_used

        logging.info(f'Calling function {function_name} with arguments {arguments}')
        function_response = await self.plugin_manager.call_function(function_name, self, arguments)

        if function_name not in plugins_used:
            plugins_used += (function_name,)

        if is_direct_result(function_response):
            self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name,
                                                content=json.dumps({'result': 'Done, the content has been sent to the user.'}))
            return function_response, plugins_used

        self.__add_function_call_to_history(chat_id=chat_id, function_name=function_name, content=function_response)
        
        # Make another API call with the function response
        common_args = {
            'temperature': self.config['temperature'],
            'max_tokens': self.config['max_tokens'],
            'stream': stream
        }
        
        if self.provider_name == "openai":
            common_args.update({
                'n': self.config['n_choices'],
                'presence_penalty': self.config['presence_penalty'],
                'frequency_penalty': self.config['frequency_penalty'],
            })
            
            functions = self.plugin_manager.get_functions_specs()
            if len(functions) > 0:
                common_args['tools'] = [{"type": "function", "function": func} for func in functions]
                common_args['tool_choice'] = 'auto' if times < self.config['functions_max_consecutive_calls'] else 'none'
        
        response = await self.provider.chat_completion(self.conversations[chat_id], **common_args)
        return await self.__handle_function_call(chat_id, response, stream, times + 1, plugins_used)

    def __add_function_call_to_history(self, chat_id, function_name, content):
        """
        Adds a function call to the conversation history
        """
        self.conversations[chat_id].append({"role": "function", "name": function_name, "content": content})

    # Image and audio methods
    async def generate_image(self, prompt: str) -> str:
        """Generate image from text prompt"""
        return await self.provider.generate_image(prompt)

    async def transcribe_audio(self, audio_file) -> str:
        """Transcribe audio to text"""
        return await self.provider.transcribe_audio(audio_file)

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech"""
        return await self.provider.text_to_speech(text)

    async def interpret_image(self, chat_id, fileobj, prompt=None):
        """Interpret image using vision model"""
        if not supports_vision(self.config['model']):
            raise NotImplementedError(f"Model {self.config['model']} doesn't support vision")
        
        image = encode_image(fileobj)
        prompt = self.config.get('vision_prompt', 'What is in this image?') if prompt is None else prompt

        content = [
            {'type': 'text', 'text': prompt}, 
            {'type': 'image_url', 'image_url': {'url': image, 'detail': self.config.get('vision_detail', 'auto')}}
        ]

        # Temporarily set vision mode
        self.conversations_vision[chat_id] = True
        
        try:
            response = await self.provider.chat_completion(
                [{"role": "user", "content": content}],
                temperature=self.config['temperature'],
                max_tokens=self.config.get('vision_max_tokens', 300)
            )
            
            answer = self._extract_response_content(response)
            
            if self.config.get('enable_vision_follow_up_questions', True):
                self.__add_to_history(chat_id, role="user", content=content)
                self.__add_to_history(chat_id, role="assistant", content=answer)
            
            tokens_used = self._get_tokens_used(response)
            
            if self.config['show_usage']:
                answer += f"\n\n---\n💰 {str(tokens_used)} {localized_text('stats_tokens', self.config['bot_language'])}"
            
            return answer, str(tokens_used)
            
        finally:
            if not self.config.get('enable_vision_follow_up_questions', True):
                self.conversations_vision[chat_id] = False