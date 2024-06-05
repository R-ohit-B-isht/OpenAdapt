"""Adapter for GPT4-V API.

https://platform.openai.com/docs/guides/vision
"""

from copy import deepcopy
from pprint import pformat
from typing import Any

from loguru import logger
from PIL import Image
import requests

from openadapt import cache, utils
from openadapt.config import config

MODEL_NAME = [
    "gpt-4-vision-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o",
][-1]
# TODO XXX: per model
MAX_TOKENS = 4096
# TODO XXX undocumented
MAX_IMAGES = None


def create_payload(
    prompt: str,
    system_prompt: str | None = None,
    images: list[Image.Image] | None = None,
    model: str = MODEL_NAME,
    detail: str = "high",  # "low" or "high"
    max_tokens: int | None = None,
) -> dict:
    """Create payload for prompting.

    Args:
        prompt: the prompt
        system_prompt: the system prompt
        images: list of images
        model: name of OpenAI model
        detail: detail level of images, "low" or "high"
        max_tokens: maximum number of tokens

    Returns:
        dict containing prompt payload
    """
    max_tokens = max_tokens or MAX_TOKENS
    if max_tokens > MAX_TOKENS:
        logger.warning(f"{max_tokens=} {MAX_TOKENS=}")
        max_tokens = MAX_TOKENS

    """Creates the payload for the API request."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    images = images or []
    for image in images:
        base64_image = utils.image2utf8(image)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_image,
                    "detail": detail,
                },
            }
        )

    if system_prompt:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            }
        ] + messages

    rval = {
        "model": model,
        "messages": messages,
    }
    if max_tokens:
        rval["max_tokens"] = max_tokens
    return rval


@cache.cache()
def get_response(
    payload: dict,
    api_key: str = config.OPENAI_API_KEY,
) -> requests.Response:
    """Sends a request to the OpenAI API and returns the response.

    Args:
        payload: dictionary returned by create_payload
        api_key (str): api key

    Returns:
        response from OpenAI API
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    return response


def get_completion(payload: dict, dev_mode: bool = False) -> str:
    """Sends a request to the OpenAI API and returns the first message.

    Args:
        payload (dict): dictionary returned by create_payload
        dev_mode (bool): whether to launch a debugger on error

    Returns:
        (str) first message from the response
    """
    response = get_response(payload)
    result = response.json()
    logger.info(f"result=\n{pformat(result)}")
    if "error" in result:
        error = result["error"]
        message = error["message"]
        # TODO: fail after maximum number of attempts
        if "retry your request" in message:
            return get_completion(payload)
        elif dev_mode:
            import ipdb

            ipdb.set_trace()
            # TODO: handle more errors
        else:
            raise ValueError(result["error"]["message"])
    choices = result["choices"]
    choice = choices[0]
    message = choice["message"]
    content = message["content"]
    return content


def log_payload(payload: dict[Any, Any]) -> None:
    """Logs a payload after removing base-64 encoded values recursively."""
    # TODO: detect base64 encoded strings dynamically
    # messages["content"][{"image_url": ...
    # payload["messages"][1]["content"][9]["image_url"]
    payload_copy = deepcopy(payload)
    for message in payload_copy["messages"]:
        for content in message["content"]:
            if "image_url" in content:
                content["image_url"]["url"] = "[REDACTED]"
    logger.info(f"payload=\n{pformat(payload_copy)}")


def prompt(
    prompt: str,
    system_prompt: str | None = None,
    images: list[Image.Image] | None = None,
    max_tokens: int | None = None,
    detail: str = "high",
) -> str:
    """Get prompt completion from OpenAI.

    Args:
        prompt: the prompt
        system_prompt: the system prompt
        images: list of images
        model: name of OpenAI model
        detail: detail level of images, "low" or "high"
        max_tokens: maximum number of tokens

    Returns:
        string containing the first message from the response
    """
    payload = create_payload(
        prompt,
        system_prompt,
        images,
        max_tokens=max_tokens,
        detail=detail,
    )
    log_payload(payload)
    result = get_completion(payload)
    logger.info(f"result=\n{pformat(result)}")
    return result
