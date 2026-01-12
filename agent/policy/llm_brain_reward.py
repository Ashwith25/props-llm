import gymnasium as gym
import random
import numpy as np
import os
import time
from jinja2 import Template
from openai import OpenAI
import google.generativeai as genai
import anthropic
import time
import socket
from stats.inverted_double_pendulum.idp_stats import evaluate_params
import json
from configs.inverted_double_pendulum.idp_summarise_template import TEMPLATE
from pydantic import BaseModel, Field, ValidationError
from typing import Tuple, Any, List, Optional, Dict
from enum import Enum
# from ollama_config import ollama_base_url
# from ollama import chat

class Operation(str, Enum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"

class Insight(BaseModel):
    operation: Operation = Field(description="Operation to be performed on the insight: create, modify, or delete.")
    insightId: Optional[int] = Field(default=None, description="Unique identifier for the insight to be modified/deleted. DO NOT provide this field when creating a new insight.")
    description: str = Field(description="Well-explained description of the insight. Leave this field empty when deleting an insight.")

class OutputSchema(BaseModel):
    reward: float = Field(description="Value of the reward aiming to estimate the most approximate reward. Please propose reward with 2 decimal places.")
    confidence: int = Field(description="Confidence score between 1 to 10, indicating your confidence on the reward your suggested.")
    reason: str = Field(description="Detailed explanation of why you chose that reward.")
    insight: Optional[Insight] = Field(default=None)

class LLMBrainReward:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        response_schema_dict = OutputSchema.model_json_schema()
        self.response_schema_json = json.dumps(response_schema_dict, indent=2)
        self.insights = []

        assert llm_model_name in [
            "o1-preview",
            "gpt-4o",
            "gemini-2.0-flash-exp",
            "gpt-4o-mini",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "o3-mini-2025-01-31",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "claude-3-7-sonnet-20250219",
            "gpt-oss:120b",
        ]
        self.llm_model_name = llm_model_name
        if "gemini" in llm_model_name:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif "claude" in llm_model_name:
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            self.model_group = "openai"
            if self.llm_model_name == 'gpt-oss:120b':
                host_node = socket.gethostname()
                asurite_id = "apoojar4"
                # print(socket.gethostbyname(host_node))
                # print(ollama_base_url())
                
                # Get the dynamic port from the environment, default to 11434 if not set
                ollama_port = os.environ.get("OLLAMA_PORT", "11434")
                
                print(f"Connecting to Ollama on {host_node}:{ollama_port}")
                
                self.client = OpenAI(
                    base_url=f"http://{asurite_id}@{host_node}:{ollama_port}/v1",  # Local Ollama API
                    api_key="ollama"              
                )
                # print(f"http://{asurite_id}@{host_node}:11434/v1")
            # if self.llm_model_name == 'gpt-oss:120b':
            #     host_node = socket.gethostname()
            #     ollama_host = os.environ.get('OLLAMA_HOST', f'{host_node}:11434')
            #     asurite_id = "apoojar4"
            #     print(f"http://{asurite_id}@{ollama_host}/v1")
            #     # print(socket.gethostbyname(host_node))
            #     # print(ollama_base_url())
            #     self.client = OpenAI(
            #         base_url=f"http://{asurite_id}@{ollama_host}/v1",  # Local Ollama API
            #         api_key="ollama"              
            #     )
            #     # print(f"http://{asurite_id}@{host_node}:11434/v1")
            else:
                self.client = OpenAI()

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role, isTool = False, body = None):
        # if self.model_group == "gpt-oss":
        #     message = {"role": role, "content": [text]}
        #     if tool_name is not None:
        #         message["tool_name"] = tool_name
        #     self.llm_conversation.append(message)
        if isTool:
            self.llm_conversation.append(body)
        elif self.model_group == "openai":
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})
    
    def query_reasoning_llm(self, content):
        # TODO: Hardcoded for OpenAI for now

        completion = self.client.chat.completions.create(
            model=self.llm_model_name,
            extra_body={"reasoning_effort": "high"},
            messages=[{
            "role": "user",
            "content": content
        }],
        )
        return completion.choices[0].message.content

    def query_llm(self):
        max_iter = [0, []]
        thinking = ""
        for attempt in range(10):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        extra_body={"reasoning_effort": "high"},
                    )
                    response = completion.choices[0].message.content
                    thinking = completion.choices[0].message.to_dict().get("reasoning", "")
                    self.add_llm_conversation(response, "assistant")

                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=1024,
                    )
                    response = message.content[0].text
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    chat_session = model.start_chat(history=self.llm_conversation[:-1])
                    response = chat_session.send_message(
                        self.llm_conversation[-1]["parts"]
                    )
                    response = response.text
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 9:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)

            if self.model_group == "openai":
                # add the response to self.llm_conversation
                self.add_llm_conversation(response, "assistant")
            else:
                self.add_llm_conversation(response, "model")

            return response, thinking

    def formatted_insights(self) -> str:
        if not self.insights:
            return "\nNo insights available."
        insights_str = ""
        for idx, insight in enumerate(self.insights):
            insights_str += f"{idx + 1}. {insight}\n"
        return insights_str

    def llm_update_parameters_num_optim_semantics(
        self,
        params: np.ndarray,
        episode_reward_buffer,
        step_number,
        env_desc_file,
        reward_range=None,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "env_description": env_desc_file,
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
                "current_iteration": step_number,
                "response_schema": self.response_schema_json,
                "target_params_string": np.array2string(params, separator=', '),
                "reward_range": reward_range,
                "memory_string": self.formatted_insights(),
            }
        )

        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        response, thinking = self.query_llm()
        api_time = time.time() - api_start_time
        try:
            validated_response = OutputSchema.model_validate_json(response)
        except ValidationError as e:
            print("INCORRECT Response from LLM:", response)
            print("Validation error:", e)
            raise e

        if validated_response.insight is not None:
            print("*"*50)
            print("Insight received:", validated_response.insight)
            print("*"*50)
            operation = validated_response.insight.operation
            if operation == Operation.CREATE:
                self.insights.append(validated_response.insight.description)
            elif operation == Operation.MODIFY:
                if validated_response.insight.insightId is not None and 0 < validated_response.insight.insightId <= len(self.insights):
                    self.insights[int(validated_response.insight.insightId) - 1] = validated_response.insight.description
            elif operation == Operation.DELETE:
                if validated_response.insight.insightId is not None and 0 < validated_response.insight.insightId <= len(self.insights):
                    del self.insights[int(validated_response.insight.insightId) - 1]
            # self.memory.append(validated_response.insight)

        # print(system_prompt)

        with open(f"memory_log.txt", "w") as memory_log_file:
            memory_log_file.write(self.formatted_insights())

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()

        # new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        # explanation = self.query_reasoning_llm(new_parameters_lis, stats) if summary else None

        return (
            validated_response.reward,
            validated_response.confidence,
            validated_response.reason,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + response
            + "\n\n\nThinking:\n"
            + thinking,
            api_time,
        )
