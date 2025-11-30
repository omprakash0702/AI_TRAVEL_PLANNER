from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from src.config.config import GROQ_API_KEY

# Initialize the LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# Define the prompt template correctly using 'messages'
itinerary_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=(
                    "You are a helpful travel assistant. Create a day trip itinerary for {city} "
                    "based on user's interests: {interests}. Provide a brief, bulleted itinerary."
                ),
                input_variables=["city", "interests"]
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="Create an itinerary for my day trip.",
                input_variables=[]
            )
        )
    ]
)

# Function to generate itinerary
def generate_itinerary(city: str, interests: list[str]) -> str:
    formatted_messages = itinerary_prompt.format_messages(
        city=city,
        interests=', '.join(interests)
    )
    response = llm.invoke(formatted_messages)
    return response.content
