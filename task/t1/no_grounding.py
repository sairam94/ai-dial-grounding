import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task import user_client
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    azure_api_key=API_KEY,
    api_version="",
)

token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    context_str = ""
    for user in context:
        context_str += f"User:\n"
        for key, value in user.items():
            context_str += f"  {key}: {value}\n"
        context_str += "\n"
    return context_str


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    
    message = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    response = await llm_client.ainvoke(messages=message)

    token_total = response.metadata['token_usage']['total_tokens'] 

    token_tracker.add_tokens(token_total)

    print(f"Response: \n {response.content}\nTokens used: {total_tokens}\n")
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        users = user_client.get_all_users()
        user_batches = [users[i:i + 100] for i in range(0, len(users), 100)]
        
        tasks = [
            generate_response(
                system_prompt=BATCH_SYSTEM_PROMPT,
                user_message=USER_PROMPT.format(
                    context=join_context(batch),
                    query=user_question
                )
            )
            for batch in user_batches
        ]

        batch_results = await asyncio.gather(*tasks)

        print("\n--- Compiling results ---")

        filtered_results = [res for res in batch_results if res != "NO_MATCHES_FOUND"]

        print(f"\n=== SEARCH RESULTS ===")

        if filtered_results:
            combined_results = "\n\n".join(filtered_results)

            await generate_response(
                system_prompt=FINAL_SYSTEM_PROMPT,
                user_message=f"SEARCH RESULTS:\n{combined_results}\n\nORIGINAL QUERY: {user_question}"
            )

        else:
            print(f"\n=== SEARCH RESULTS ===")
            print(f"No users found matching '{user_question}'")
            print("\nTry refining your search or using different keywords.")

        usage_summary = token_tracker.get_summary()
        print(f"\n--- Token Usage Summary ---")
        print(f"Total tokens used: {usage_summary['total_tokens']}")
        print(f"Number of batches processed: {usage_summary['batch_count']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation