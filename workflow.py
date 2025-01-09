from typing import Annotated, List, Sequence
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm import llm

MAX_REVISION_TIMES = 5

class State(TypedDict):
    messages: Annotated[list, add_messages]
    revision_times: int

class Writer:
    def __init__(self, llm=llm):
        self.llm = llm
        workflow = StateGraph(State)

        # Define the workflow nodes and edges
        workflow.add_node('writer', self.writer_call)
        workflow.add_edge(START, 'writer')
        workflow.add_conditional_edges(
            'writer', 
            self.writer_condition, 
            {True: 'critic', False: END}  # Skip profiles that don't match criteria
        )
        workflow.add_node('critic', self.critic_call)
        workflow.add_conditional_edges(
            'critic', 
            self.critic_condition, 
            {True: END, False: END}  # End if critic feedback fails
        )

        memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=memory)

    def writer_call(self, state):
        """
        LLM call to validate profile against criteria.
        """
        system_prompt = """
        You are a professional Talent Acquisition Specialist. 
        Your task is to evaluate a LinkedIn profile description based on the provided criteria.
        If the profile is an excellent match, respond with 'Valid profile'.
        Otherwise, respond with 'Invalid profile'.
        """
        profile_description = state['messages'][0]
        criteria = state.get('criteria', "No specific criteria provided.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Criteria: {criteria}"},
            {"role": "user", "content": f"Profile Description:\n{profile_description}"}
        ]
        
        response = self.llm.call(messages)
        valid_profile = response.strip() == "Valid profile"
        return {'messages': [response], 'valid_profile': valid_profile}

    def writer_condition(self, state):
        """
        Skip profiles that don't match criteria.
        """
        return state.get('valid_profile', False)

    def critic_call(self, state):
        """
        LLM call to provide detailed feedback and suggestions.
        """
        system_prompt = """
        You are a professional talent acquisition critique, given the profile, please give a feedback why it's good or not good to continue with the profile.
        """
        profile_output = state['messages'][-1]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": profile_output}
        ]
        
        response = self.llm.call(messages)
        print(f"Critic Output: {response}")
        return {'messages': state['messages'] + [response]}

    def critic_condition(self, state):
        """
        Decide if the critic feedback allows continuation.
        """
        feedback = state['messages'][-1]
        return "satisfactory" in feedback.lower()  # Example condition

# Example usage
writer = Writer()
state = {'messages': ["Profile description here..."], 'criteria': "Relevant experience in AI."}
result = writer.graph.run(state)
