import httpx
import json
import logging
import asyncio
import os
import traceback
try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

class OllamaService(BaseChatModel):
    base_url: str
    username: str
    password: str
    model: str
    timeout: float = 60.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "ollama_custom"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Simple sync wrapper for agenerate
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            
        return loop.run_until_complete(self._agenerate(messages, stop, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ollama_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            ollama_messages.append({"role": role, "content": m.content})


        # If the user set the model to the sentinel value 'ollama_cloud',
        # call the Ollama Cloud API using the official client and OLLAMA_API_KEY.
        if (self.model == "ollama_cloud"):
            logger.info("🌐 Ollama Cloud API mode activated")
            
            # Validate Ollama Client availability
            if OllamaClient is None:
                error_msg = "Ollama package not installed. Install with: pip install ollama"
                logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            
            # Get and validate configuration
            # Note: Cloud models should NOT have -cloud suffix when calling the API
            # The -cloud suffix is only for local CLI usage
            cloud_model = os.getenv("OLLAMA_CLOUD_MODEL", "gpt-oss:120b")
            # Remove -cloud suffix if present (user might have it in .env from CLI usage)
            if cloud_model.endswith("-cloud"):
                cloud_model = cloud_model[:-6]  # Remove last 6 characters ("-cloud")
                logger.info(f"🔧 Removed -cloud suffix, using: {cloud_model}")
            
            api_key = os.getenv("OLLAMA_API_KEY")
            
            if not api_key:
                error_msg = "OLLAMA_API_KEY environment variable is not set. Please configure it in .env file"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Remove quotes if present in the API key
            api_key = api_key.strip('"').strip("'")
            
            logger.info(f"📡 Using Ollama Cloud model: {cloud_model}")
            logger.info(f"🔑 API Key configured: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")

            def cloud_call():
                """Synchronous wrapper for Ollama Cloud API call
                Based on verified working example from user
                """
                try:
                    # Use https://ollama.com as host (not /api/chat endpoint)
                    client = OllamaClient(
                        host="https://ollama.com",
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                    logger.info(f"🚀 Sending request to Ollama Cloud with {len(ollama_messages)} messages")
                    # Call with model name WITHOUT -cloud suffix
                    response = client.chat(model=cloud_model, messages=ollama_messages)
                    logger.info("✅ Ollama Cloud response received successfully")
                    return response
                except Exception as e:
                    logger.error(f"❌ Ollama Cloud API call error: {type(e).__name__}: {str(e)}")
                    raise

            try:
                # Execute cloud call in thread pool
                resp = await asyncio.to_thread(cloud_call)
                
                # DEBUG: Log the response object structure
                logger.debug(f"🔍 Response object type: {type(resp).__name__}")
                logger.debug(f"🔍 Response dir: {[x for x in dir(resp) if not x.startswith('_')]}")
                
                # Extract content from response
                content = None
                msg_data = getattr(resp, "message", {})
                
                # Try multiple extraction paths
                # Path 1: msg_data dict with 'content' or 'response' key
                if isinstance(msg_data, dict):
                    content = msg_data.get("content") or msg_data.get("response")
                    logger.debug(f"🔍 Path 1 (dict keys): content={content is not None}")
                
                # Path 2: msg_data object with content attribute
                if not content and hasattr(msg_data, "content"):
                    content = msg_data.content  # type: ignore
                    logger.debug(f"🔍 Path 2 (msg_data.content): content={content is not None}")
                
                # Path 3: resp.message string
                if not content and hasattr(resp, "message"):
                    content = str(resp.message)
                    logger.debug(f"🔍 Path 3 (str(resp.message)): content={content is not None}")
                
                # Path 4: Direct resp attributes (for dict-like responses)
                if not content and isinstance(resp, dict):
                    content = resp.get("content") or resp.get("message", {}).get("content") or resp.get("response")
                    logger.debug(f"🔍 Path 4 (resp dict): content={content is not None}")
                
                # Fallback to JSON serialization if content is still a dict
                if isinstance(content, dict):
                    content = content.get("content") or content.get("response") or json.dumps(content)
                
                if not content:
                    logger.warning(f"⚠️ No content extracted from Ollama Cloud response. resp type={type(resp).__name__}, msg_data={msg_data}")
                    logger.warning(f"⚠️ Full response object: {resp}")
                    content = ""
                
                logger.info(f"📝 Extracted content length: {len(str(content))} characters")

                # Extract tool_calls if present
                raw_tool_calls = []
                if isinstance(msg_data, dict):
                    raw_tool_calls = msg_data.get('tool_calls', [])
                elif hasattr(msg_data, 'tool_calls'):
                    raw_tool_calls = msg_data.tool_calls or []

                formatted_tool_calls = []
                for i, tc in enumerate(raw_tool_calls):
                    try:
                        if "function" in tc:
                            f = tc["function"]
                            args = f.get("arguments")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    logger.warning(f"⚠️ Failed to parse tool call arguments as JSON: {args}")
                            
                            formatted_tool_calls.append({
                                "name": f.get("name"),
                                "args": args,
                                "id": tc.get("id", f"call_{i}_{id(tc)}")
                            })
                        else:
                            formatted_tool_calls.append(tc)
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing tool call {i}: {e}")
                        continue
                
                if formatted_tool_calls:
                    logger.info(f"🔧 Extracted {len(formatted_tool_calls)} tool calls")

                ai_message = AIMessage(content=content or "", tool_calls=formatted_tool_calls)
                logger.info("✅ Ollama Cloud request completed successfully")
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
                
            except ValueError as ve:
                # Configuration errors
                logger.error(f"❌ Configuration error: {ve}")
                raise
            except ConnectionError as ce:
                # Network errors
                logger.error(f"❌ Network error connecting to Ollama Cloud: {ce}")
                raise RuntimeError(f"Failed to connect to Ollama Cloud API: {ce}")
            except Exception as e:
                # Generic errors
                logger.error(f"❌ Ollama Cloud call failed: {type(e).__name__}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Ollama Cloud API error: {str(e)}")

        # Process tools from kwargs if they come from bind_tools or elsewhere
        tools = kwargs.pop("tools", None)
        
        # Process format from kwargs (common for JSON enforcement)
        format_val = kwargs.pop("format", None)
        
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            **kwargs
        }
        
        if format_val:
            payload["format"] = format_val
        
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = None
            try:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    auth=(self.username, self.password),
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    err_msg = f"Ollama HTTP Error {response.status_code}: {response.text[:500]}"
                    logger.error(err_msg)
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=err_msg))])

                data = response.json()
                msg_data = data.get('message', {})
                content = msg_data.get('content', data.get('response', ''))
                
                # Handle tool_calls if present in response
                raw_tool_calls = msg_data.get('tool_calls', [])
                formatted_tool_calls = []
                for i, tc in enumerate(raw_tool_calls):
                    if "function" in tc:
                        f = tc["function"]
                        args = f.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                pass
                        
                        formatted_tool_calls.append({
                            "name": f.get("name"),
                            "args": args,
                            "id": tc.get("id", f"call_{i}_{id(tc)}")
                        })
                    else:
                        formatted_tool_calls.append(tc)

                ai_message = AIMessage(content=content, tool_calls=formatted_tool_calls)
                
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            except json.JSONDecodeError:
                raw = response.text[:500] if response else "No response"
                logger.error(f"Ollama JSON Decode Error. Raw response: {raw}")
                raise
            except Exception as e:
                logger.error(f"Ollama _agenerate Error: {e}")
                raise

    def bind_tools(
        self,
        tools: Union[List[Any], Any],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """
        Bind tools for compatibility with LangGraph and other LangChain components.
        """
        from langchain_core.utils.function_calling import convert_to_openai_function
        openai_tools = [convert_to_openai_function(t) for t in tools]
        ollama_tools = [{"type": "function", "function": t} for t in openai_tools]
        
        return self.bind(tools=ollama_tools, **kwargs)

    async def close(self):
        pass
