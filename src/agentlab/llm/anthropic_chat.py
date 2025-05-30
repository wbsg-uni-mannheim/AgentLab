"""
Anthropic Chat API integration for AgentLab.
This module provides Anthropic Claude model support while keeping changes separate 
from the main AgentLab codebase for easier updates.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

import agentlab.llm.tracking as tracking
from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.llm_utils import AIMessage


class RetryError(Exception):
    """Exception raised when API retries are exhausted."""
    pass


@dataclass
class AnthropicModelArgs(BaseModelArgs):
    """Serializable object for instantiating a generic chat model with an Anthropic model."""

    def make_model(self):
        return AnthropicChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            log_probs=self.log_probs,
        )


class AnthropicChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name,
        api_key=None,
        temperature=0.5,
        max_tokens=100,
        max_retry=4,
        min_retry_wait_time=60,
        log_probs=False,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed. Please install it with: pip install anthropic")
        
        assert max_retry > 0, "max_retry should be greater than 0"

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.log_probs = log_probs

        # Get the API key from the environment variable if not provided
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.api_key = api_key

        # Get pricing information
        try:
            from agentlab.llm.anthropic_pricing import get_pricing_anthropic
            pricings = get_pricing_anthropic()
            self.input_cost = float(pricings[model_name]["prompt"])
            self.output_cost = float(pricings[model_name]["completion"])
        except (ImportError, KeyError, TypeError):
            logging.warning(
                f"Model {model_name} not found in the pricing information, prices are set to 0."
            )
            self.input_cost = 0.0
            self.output_cost = 0.0

        self.client = anthropic.Anthropic(api_key=api_key)

    def _convert_messages_to_anthropic_format(self, messages: list[dict]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system_message = ""
        converted_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                system_message = content
            elif role == "user":
                converted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                converted_messages.append({"role": "assistant", "content": content})
        
        return system_message, converted_messages

    def __call__(self, messages: list[dict], n_samples: int = 1, temperature: float = None) -> dict:
        # Initialize retry tracking attributes
        self.retries = 0
        self.success = False
        self.error_types = []

        if n_samples > 1:
            raise NotImplementedError("Multiple samples not supported for Anthropic models yet")

        completion = None
        e = None
        temperature = temperature if temperature is not None else self.temperature
        
        # Convert messages to Anthropic format
        system_message, anthropic_messages = self._convert_messages_to_anthropic_format(messages)

        for itr in range(self.max_retry):
            self.retries += 1
            try:
                completion = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    system=system_message if system_message else None,
                    messages=anthropic_messages,
                )

                self.success = True
                break
            except anthropic.APIError as e:
                error_type = self._handle_anthropic_error(e, itr)
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                f"Failed to get a response from the Anthropic API after {self.max_retry} retries\n"
                f"Last error: {self.error_types[-1] if self.error_types else 'Unknown error'}"
            )

        input_tokens = completion.usage.input_tokens
        output_tokens = completion.usage.output_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        res = AIMessage(completion.content[0].text)
        if self.log_probs:
            # Anthropic doesn't provide log_probs in the same way as OpenAI
            res["log_probs"] = None
        return res

    def _handle_anthropic_error(self, error, itr):
        """Handle Anthropic API errors with retry logic."""
        logging.warning(
            f"Failed to get a response from Anthropic API: \n{error}\n" 
            f"Retrying... ({itr+1}/{self.max_retry})"
        )
        
        # For rate limiting, Anthropic uses different error types
        if hasattr(error, 'status_code') and error.status_code == 429:
            wait_time = self.min_retry_wait_time
        else:
            wait_time = self.min_retry_wait_time
            
        logging.info(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
        return str(error)

    def get_stats(self):
        return {
            "n_retry_llm": self.retries,
        }