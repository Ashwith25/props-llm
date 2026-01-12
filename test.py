from agent.policy.llm_brain_reward import OutputSchema
import json

response_schema_dict = OutputSchema.model_json_schema()
response_schema_json = json.dumps(response_schema_dict, indent=2)

print(response_schema_json)