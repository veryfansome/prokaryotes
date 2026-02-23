from openai.types.responses.response_input_param import FunctionCallOutput

from prokaryotes.llm_v1 import FunctionToolCallback

class SearchEmailFunctionToolCallback(FunctionToolCallback):
    async def call(self, arguments: str, call_id: str) -> FunctionCallOutput:
        return FunctionCallOutput(
            type="function_call_output",
            call_id=call_id,
            output='{"messages": []}'
        )
