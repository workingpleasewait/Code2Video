import openai
import time
import random
import os
import base64
from openai import OpenAI
import time
import json
import pathlib
import anthropic
from google import genai
import urllib.request
import urllib.error


# Read and cache once
_CFG_PATH = pathlib.Path(__file__).with_name("api_config.json")
with _CFG_PATH.open("r", encoding="utf-8") as _f:
    _CFG = json.load(_f)


def cfg(svc: str, key: str, default=None):
    return os.getenv(f"{svc}_{key}".upper(), _CFG.get(svc, {}).get(key, default))


def generate_log_id():
    """Generate a log ID with 'tkb' prefix and current timestamp."""
    return f"tkb{int(time.time() * 1000)}"


def request_claude(prompt, log_id=None, max_tokens=16384, max_retries=3):
    """Claude via Anthropic Messages API (no gateway required).
    Returns text string content.
    """
    import anthropic
    # Prefer explicit Anthropic key; fall back to CLAUDE_API_KEY from cfg/env
    api_key = os.getenv("ANTHROPIC_API_KEY") or cfg("claude", "api_key")
    model = os.getenv("CLAUDE_MODEL") or cfg("claude", "model", "claude-3.5-sonnet-latest")

    if not api_key:
        raise RuntimeError("Anthropic API key missing: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)

    if log_id is None:
        log_id = generate_log_id()

    retry_count = 0
    while retry_count < max_retries:
        try:
            effective_max = min(int(max_tokens or 1000), int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "4000")))
            completion = client.messages.create(
                model=model,
                max_tokens=effective_max,
                messages=[{"role": "user", "content": prompt}],
            )
            # Concatenate text blocks
            def _wrap_json_if_present(text: str) -> str:
                if "```" in text:
                    return text
                start = text.find("{")
                if start != -1:
                    depth = 0
                    end = None
                    for i, ch in enumerate(text[start:], start):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if end is not None and end > start:
                        json_str = text[start : end + 1]
                        return f"```json\n{json_str}\n```"
                return text

            text_out_raw = "\n".join(
                [blk.text for blk in getattr(completion, "content", []) if getattr(blk, "type", None) == "text"]
            ).strip()
            text_out = _wrap_json_if_present(text_out_raw)
            return text_out
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)


def request_claude_token(prompt, log_id=None, max_tokens=10000, max_retries=3):
    """Claude via Anthropic Messages API (returns completion-like object and usage).
    Shapes the response to be compatible with downstream parsing:
    - response.candidates[0].content.parts[0].text
    - response.choices[0].message.content
    """
    import anthropic

    # Prefer explicit Anthropic key; fall back to CLAUDE_API_KEY from cfg/env
    api_key = os.getenv("ANTHROPIC_API_KEY") or cfg("claude", "api_key")
    model = os.getenv("CLAUDE_MODEL") or cfg("claude", "model", "claude-3.5-sonnet-latest")

    if not api_key:
        raise RuntimeError("Anthropic API key missing: set ANTHROPIC_API_KEY or CLAUDE_API_KEY")

    class _DummyPart:
        def __init__(self, text):
            self.text = text
    class _DummyContent:
        def __init__(self, text):
            self.parts = [_DummyPart(text)]
    class _DummyCandidate:
        def __init__(self, text):
            self.content = _DummyContent(text)
    class _DummyMessage:
        def __init__(self, text):
            self.content = text
    class _DummyChoice:
        def __init__(self, text):
            self.message = _DummyMessage(text)
    class _DummyCompletion:
        def __init__(self, text, usage):
            self.candidates = [_DummyCandidate(text)]
            self.choices = [_DummyChoice(text)]
            self.usage = usage

    client = anthropic.Anthropic(api_key=api_key)

    if log_id is None:
        log_id = generate_log_id()

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    retry_count = 0
    while retry_count < max_retries:
        try:
            effective_max = min(int(max_tokens or 1000), int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "4000")))
            completion = client.messages.create(
                model=model,
                max_tokens=effective_max,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text
            def _wrap_json_if_present(text: str) -> str:
                if "```" in text:
                    return text
                start = text.find("{")
                if start != -1:
                    depth = 0
                    end = None
                    for i, ch in enumerate(text[start:], start):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if end is not None and end > start:
                        json_str = text[start : end + 1]
                        return f"```json\n{json_str}\n```"
                return text

            text_out_raw = "\n".join(
                [blk.text for blk in getattr(completion, "content", []) if getattr(blk, "type", None) == "text"]
            ).strip()
            text_out = _wrap_json_if_present(text_out_raw)

            # Usage mapping
            if getattr(completion, "usage", None):
                input_tokens = getattr(completion.usage, "input_tokens", 0)
                output_tokens = getattr(completion.usage, "output_tokens", 0)
                usage_info["prompt_tokens"] = input_tokens
                usage_info["completion_tokens"] = output_tokens
                usage_info["total_tokens"] = input_tokens + output_tokens

            dummy = _DummyCompletion(text_out, usage_info)
            return dummy, usage_info

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)

    return None, usage_info


def request_gemini_with_video(prompt: str, video_path: str, log_id=None, max_tokens: int = 10000, max_retries: int = 3):
    """
    Makes a multimodal request to the Gemini-2.5 model using video + text via Google GenAI client.
    Returns the raw response object.
    """
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or cfg("gemini", "api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    )
    model_name = os.getenv("GEMINI_MODEL") or cfg("gemini", "model", "gemini-2.5-pro-preview-05-06")
    if not api_key:
        raise RuntimeError("Gemini API key missing: set GEMINI_API_KEY or GOOGLE_API_KEY or config gemini.api_key")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    client = genai.Client(api_key=api_key)

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.models.generate_content(
                model=model_name,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                        ],
                    }
                ],
                config={"max_output_tokens": min(int(max_tokens or 1000), int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4000")))}
            )
            return completion
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.2 + random.random() * 0.2
            print(f"Retry {retry_count}/{max_retries} after error: {e}, waiting {delay:.2f}s...")
            time.sleep(delay)


def request_gemini_video_img(
    prompt: str, video_path: str, image_path: str, log_id=None, max_tokens: int = 10000, max_retries: int = 3
):
    """
    Makes a multimodal request to the Gemini-2.5 model using video & ref img + text via Google GenAI client.
    Returns (response, usage_info) to match agent usage tracking.
    """
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or cfg("gemini", "api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    )
    model_name = os.getenv("GEMINI_MODEL") or cfg("gemini", "model", "gemini-2.5-pro-preview-05-06")
    if not api_key:
        raise RuntimeError("Gemini API key missing: set GEMINI_API_KEY or GOOGLE_API_KEY or config gemini.api_key")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    client = genai.Client(api_key=api_key)

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.models.generate_content(
                model=model_name,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "video/mp4", "data": video_b64}},
                            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                        ],
                    }
                ],
                config={"max_output_tokens": min(int(max_tokens or 1000), int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4000")))}
            )
            # Map usage
            um = getattr(completion, "usage_metadata", None)
            if um is None and isinstance(completion, dict):
                um = completion.get("usage_metadata")
            if um is not None:
                try:
                    pt = getattr(um, "prompt_token_count", None)
                    ct = getattr(um, "candidates_token_count", None)
                    tt = getattr(um, "total_token_count", None)
                    if pt is None and isinstance(um, dict):
                        pt = um.get("prompt_token_count")
                        ct = um.get("candidates_token_count")
                        tt = um.get("total_token_count")
                    usage_info["prompt_tokens"] = int(pt or 0)
                    usage_info["completion_tokens"] = int(ct or 0)
                    usage_info["total_tokens"] = int(tt or ((pt or 0) + (ct or 0)))
                except Exception:
                    pass
            return completion, usage_info
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.2 + random.random() * 0.2
            print(f"Retry {retry_count}/{max_retries} after error: {e}, waiting {delay:.2f}s...")
            time.sleep(delay)
    return None, usage_info


def request_gemini_video_img_token(
    prompt: str, video_path: str, image_path: str, log_id=None, max_tokens: int = 10000, max_retries: int = 3
):
    """
    Makes a multimodal request to the Gemini-2.5 model using video & ref img + text.

    Args:
        prompt (str): The user instruction, e.g., "Please evaluate and suggest improvements for this educational animation."
        video_path (str): Local path to the video file (MP4 preferred, <20MB recommended).
        log_id (str, optional): Tracking ID
        max_tokens (int): Max response token length
        max_retries (int): Max retry attempts

    Returns:
        dict: The Gemini model response
    """
    base_url = cfg("gemini", "base_url")
    api_version = cfg("gemini", "api_version")
    api_key = cfg("gemini", "api_key")
    model_name = cfg("gemini", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=api_key,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Load and base64-encode video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    video_data_url = f"data:video/mp4;base64,{video_base64}"

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_data_url = f"data:image/png;base64,{base64_image}"

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": video_data_url, "detail": "high"},
                                "media_type": "video/mp4",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url, "detail": "high"},
                                "media_type": "image/png",
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )
            # return completion

            if completion.usage:
                usage_info["prompt_tokens"] = completion.usage.prompt_tokens
                usage_info["completion_tokens"] = completion.usage.completion_tokens
                usage_info["total_tokens"] = completion.usage.total_tokens
            return completion, usage_info

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.2 + random.random() * 0.2
            print(f"Retry {retry_count}/{max_retries} after error: {e}, waiting {delay:.2f}s...")
            time.sleep(delay)
    return None, usage_info


def request_gemini(prompt, log_id=None, max_tokens=8000, max_retries=3):
    """
    Text request to Gemini model via Google GenAI client.
    Returns the raw response.
    """
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or cfg("gemini", "api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    )
    model_name = os.getenv("GEMINI_MODEL") or cfg("gemini", "model", "gemini-2.5-pro-preview-05-06")
    if not api_key:
        raise RuntimeError("Gemini API key missing: set GEMINI_API_KEY or GOOGLE_API_KEY or config gemini.api_key")

    client = genai.Client(api_key=api_key)

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"max_output_tokens": min(int(max_tokens or 1000), int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4000")))}
            )
            return completion
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)


def request_gemini_token(prompt, log_id=None, max_tokens=8000, max_retries=3):
    """
    Text request to Gemini model via Google GenAI client; returns (response, usage_info).
    """
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or cfg("gemini", "api_key")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    )
    model_name = os.getenv("GEMINI_MODEL") or cfg("gemini", "model", "gemini-2.5-pro-preview-05-06")
    if not api_key:
        raise RuntimeError("Gemini API key missing: set GEMINI_API_KEY or GOOGLE_API_KEY or config gemini.api_key")

    client = genai.Client(api_key=api_key)

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.models.generate_content(
                model=model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"max_output_tokens": min(int(max_tokens or 1000), int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4000")))}
            )
            um = getattr(completion, "usage_metadata", None)
            if um is None and isinstance(completion, dict):
                um = completion.get("usage_metadata")
            if um is not None:
                try:
                    pt = getattr(um, "prompt_token_count", None)
                    ct = getattr(um, "candidates_token_count", None)
                    tt = getattr(um, "total_token_count", None)
                    if pt is None and isinstance(um, dict):
                        pt = um.get("prompt_token_count")
                        ct = um.get("candidates_token_count")
                        tt = um.get("total_token_count")
                    usage_info["prompt_tokens"] = int(pt or 0)
                    usage_info["completion_tokens"] = int(ct or 0)
                    usage_info["total_tokens"] = int(tt or ((pt or 0) + (ct or 0)))
                except Exception:
                    pass
            return completion, usage_info
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)
    return None, usage_info


def request_ollama_token(prompt, log_id=None, max_tokens=4000, max_retries=1):
    """
    Local free text-generation via Ollama chat API.
    - Uses OLLAMA_MODEL (default: qwen2.5-coder:3b) and OLLAMA_BASE (default: http://localhost:11434)
    - Returns a Claude/Gemini-like completion object and usage dict to fit existing parsing paths
    """
    base = os.getenv("OLLAMA_BASE", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")

    class _DummyPart:
        def __init__(self, text):
            self.text = text
    class _DummyContent:
        def __init__(self, text):
            self.parts = [_DummyPart(text)]
    class _DummyCandidate:
        def __init__(self, text):
            self.content = _DummyContent(text)
    class _DummyMessage:
        def __init__(self, text):
            self.content = text
    class _DummyChoice:
        def __init__(self, text):
            self.message = _DummyMessage(text)
    class _DummyCompletion:
        def __init__(self, text, usage):
            self.candidates = [_DummyCandidate(text)]
            self.choices = [_DummyChoice(text)]
            self.usage = usage

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    url = f"{base.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": min(int(max_tokens or 512), 1024)}
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp_text = resp.read().decode("utf-8", errors="ignore")
            try:
                obj = json.loads(resp_text)
                # Ollama chat returns final message content at message.content
                text_out = obj.get("message", {}).get("content", "") or ""
            except Exception:
                text_out = resp_text
            dummy = _DummyCompletion(text_out, usage_info)
            return dummy, usage_info
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection failed: {e.reason}")


def request_gpt4o(prompt, log_id=None, max_tokens=8000, max_retries=3):
    """
    Makes a request to the gpt-4o-2024-11-20 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 8000
        max_retries (int, optional): Maximum number of retry attempts, default 3

    Returns:
        dict: The model's response
    """

    base_url = cfg("gpt4o", "base_url")
    api_version = cfg("gpt4o", "api_version")
    ak = cfg("gpt4o", "api_key")
    model_name = cfg("gpt4o", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )
            return completion.choices[0].message.content
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

            # Exponential backoff with jitter
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)


def request_gpt4o_token(prompt, log_id=None, max_tokens=8000, max_retries=3):
    """
    Makes a request to the gpt-4o-2024-11-20 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 8000
        max_retries (int, optional): Maximum number of retry attempts, default 3

    Returns:
        dict: The model's response
    """
    base_url = cfg("gpt4o", "base_url")
    api_version = cfg("gpt4o", "api_version")
    ak = cfg("gpt4o", "api_key")
    model_name = cfg("gpt4o", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )

            if completion.usage:
                usage_info["prompt_tokens"] = completion.usage.prompt_tokens
                usage_info["completion_tokens"] = completion.usage.completion_tokens
                usage_info["total_tokens"] = completion.usage.total_tokens
            return completion, usage_info

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

            # Exponential backoff with jitter
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)
    return None, usage_info


def request_o4mini(prompt, log_id=None, max_tokens=8000, max_retries=3, thinking=False):
    """
    Makes a request to the o4-mini-2025-04-16 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 8000
        max_retries (int, optional): Maximum number of retry attempts, default 3
        thinking (bool, optional): Whether to enable thinking mode, default False

    Returns:
        dict: The model's response
    """
    base_url = cfg("gpt4omini", "base_url")
    api_version = cfg("gpt4omini", "api_version")
    ak = cfg("gpt4omini", "api_key")
    model_name = cfg("gpt4omini", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    # Configure extra_body for thinking if enabled
    extra_body = None
    if thinking:
        extra_body = {"thinking": {"type": "enabled", "budget_tokens": 2000}}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
            return completion
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

            # Exponential backoff with jitter
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)


def request_o4mini_token(prompt, log_id=None, max_tokens=8000, max_retries=3, thinking=False):
    """
    Makes a request to the o4-mini-2025-04-16 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 8000
        max_retries (int, optional): Maximum number of retry attempts, default 3
        thinking (bool, optional): Whether to enable thinking mode, default False

    Returns:
        dict: The model's response
    """
    base_url = cfg("gpt4omini", "base_url")
    api_version = cfg("gpt4omini", "api_version")
    ak = cfg("gpt4omini", "api_key")
    model_name = cfg("gpt4omini", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Configure extra_body for thinking if enabled
    extra_body = None
    if thinking:
        extra_body = {"thinking": {"type": "enabled", "budget_tokens": 2000}}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )

            if completion.usage:
                usage_info["prompt_tokens"] = completion.usage.prompt_tokens
                usage_info["completion_tokens"] = completion.usage.completion_tokens
                usage_info["total_tokens"] = completion.usage.total_tokens
            return completion, usage_info

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

            # Exponential backoff with jitter
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)
    return None, usage_info


def request_gpt5(prompt, log_id=None, max_tokens=1000, max_retries=3):
    """
    Makes a request to the gpt-5-chat-2025-08-07 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 1000
        max_retries (int, optional): Maximum number of retry attempts, default 3

    Returns:
        dict: The model's response
    """

    base_url = cfg("gpt5", "base_url")
    api_version = cfg("gpt5", "api_version")
    ak = cfg("gpt5", "api_key")
    model_name = cfg("gpt5", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )
            return completion
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

            # Exponential backoff with jitter
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)


def request_gpt5_token(prompt, log_id=None, max_tokens=1000, max_retries=3):
    """
    Makes a request to the gpt-5-chat-2025-08-07 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 1000
        max_retries (int, optional): Maximum number of retry attempts, default 3

    Returns:
        dict: The model's response
    """
    base_url = cfg("gpt5", "base_url")
    api_version = cfg("gpt5", "api_version")
    ak = cfg("gpt5", "api_key")
    model_name = cfg("gpt5", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )

            if completion.usage:
                usage_info["prompt_tokens"] = completion.usage.prompt_tokens
                usage_info["completion_tokens"] = completion.usage.completion_tokens
                usage_info["total_tokens"] = completion.usage.total_tokens
            return completion, usage_info

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

            # Exponential backoff with jitter
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)
    return None, usage_info


def request_gpt41(prompt, log_id=None, max_tokens=1000, max_retries=3):
    """
    Makes a request to the gpt-4.1-2025-04-14 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 1000
        max_retries (int, optional): Maximum number of retry attempts, default 3

    Returns:
        dict: The model's response
    """
    base_url = cfg("gpt41", "base_url")
    api_version = cfg("gpt41", "api_version")
    api_key = cfg("gpt41", "api_key")
    model_name = cfg("gpt41", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=api_key,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )
            return completion
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")

            # Exponential backoff with jitter
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)


def request_gpt41_token(prompt, log_id=None, max_tokens=1000, max_retries=3):
    """
    Makes a request to the gpt-4.1-2025-04-14 model with retry functionality.

    Args:
        prompt (str): The text prompt to send to the model
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 1000
        max_retries (int, optional): Maximum number of retry attempts, default 3

    Returns:
        dict: The model's response
    """
    base_url = cfg("gpt41", "base_url")
    api_version = cfg("gpt41", "api_version")
    ak = cfg("gpt41", "api_key")
    model_name = cfg("gpt41", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )

    if log_id is None:
        log_id = generate_log_id()

    extra_headers = {"X-TT-LOGID": log_id}
    usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )

            if completion.usage:
                usage_info["prompt_tokens"] = completion.usage.prompt_tokens
                usage_info["completion_tokens"] = completion.usage.completion_tokens
                usage_info["total_tokens"] = completion.usage.total_tokens
            return completion, usage_info

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                # 即使失败也返回，以便主程序可以继续
                print(f"Failed after {max_retries} attempts. Last error: {str(e)}")
                return None, usage_info

            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)

    return None, usage_info


def request_gpt41_img(prompt, image_path=None, log_id=None, max_tokens=1000, max_retries=3):
    """
    Makes a request to the gpt-4.1-2025-04-14 model with optional image input and retry functionality.
    Args:
        prompt (str): The text prompt to send to the model
        image_path (str, optional): Absolute path to an image file to include
        log_id (str, optional): The log ID for tracking requests, defaults to tkb+timestamp
        max_tokens (int, optional): Maximum tokens for response, default 1000
        max_retries (int, optional): Maximum number of retry attempts, default 3
    Returns:
        dict: The model's response
    """
    base_url = cfg("gpt41", "base_url")
    api_version = cfg("gpt41", "api_version")
    ak = cfg("gpt41", "api_key")
    model_name = cfg("gpt41", "model")

    client = openai.AzureOpenAI(
        azure_endpoint=base_url,
        api_version=api_version,
        api_key=ak,
    )
    if log_id is None:
        log_id = generate_log_id()
    extra_headers = {"X-TT-LOGID": log_id}

    if image_path:
        # 检查图片路径是否存在
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            }
        ]

    else:
        messages = [{"role": "user", "content": prompt}]
    retry_count = 0
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                extra_headers=extra_headers,
            )
            return completion
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"Failed after {max_retries} attempts. Last error: {str(e)}")
            delay = (2**retry_count) * 0.1 + (random.random() * 0.1)
            print(
                f"Request failed with error: {str(e)}. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})"
            )
            time.sleep(delay)


if __name__ == "__main__":

    # Gemini
    # response_gemini = request_gemini("上海天气怎么样？")
    # print(response_gemini.model_dump_json())

    # # GPT-4o
    # response_gpt4o = request_gpt4o("上海天气怎么样？")
    # print(response_gpt4o)

    # # o4-mini
    # response_o4mini = request_o4mini("上海天气怎么样？")
    # print(response_o4mini.model_dump_json())

    # # GPT-4.1
    response_gpt41 = request_gpt41("上海天气怎么样？")
    print(response_gpt41.model_dump_json())

    # GPT-5
    # response_gpt5 = request_gpt5("新加坡天气怎么样？")
    # print(response_gpt5.model_dump_json())

    # # Claude
    # response_claude = request_claude_token("新加坡天气怎么样？")
    # print(response_claude)
