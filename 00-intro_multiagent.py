import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.utils.logging import setup_logging

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.group_chat.agent_group_chat import AgentGroupChat
from semantic_kernel.agents.strategies.termination.default_termination_strategy import DefaultTerminationStrategy
from semantic_kernel.agents.strategies import (
    KernelFunctionTerminationStrategy,
)
import streamlit as st

from semantic_kernel.contents import ChatHistoryTruncationReducer

from semantic_kernel.functions import KernelFunctionFromPrompt

################################################
# Tracing dans le fichier telemetry : configuration des logs et connexion App Insights
from telemetry import set_up_logging, set_up_metrics, set_up_tracing

set_up_logging()
set_up_tracing()
set_up_metrics()
################################################

################################################
# Set up des angents
bootcamp_agent_name = "Organisateur"
present_agent_name = "Presentateur"

avatars = {bootcamp_agent_name: r"C:\Users\AlexandreFenneteau\Travail\bootcamp\semantic_kernel\assets\logo-organizer.png",
           present_agent_name: r"C:\Users\AlexandreFenneteau\Travail\bootcamp\semantic_kernel\assets\logo-presenter.png"}

BOOTCAMP_AGENT_INSTRUCTIONS = f"You want to make {present_agent_name} do a presentation about building AI agents using the middleware 'Semantic Kernel' only and try to convince him to do it. When {present_agent_name} conceides to do a presentation on 'Semantic Kernel' you mention that you agree by saying 'Je suis d'accord'. You stay brief in your messages, with maximum 2 sentences. You speak French."
PRESENTER_AGENT_INSTRUCTIONS = f"You want to do a presentation about building AI agents but with 'Hugging Face smolagent' framework and try to convince the other agent to do so. After 2 asking, you conceide to the view of {bootcamp_agent_name}. You stay brief in your messages, with maximum 2 sentences. You speack French."

termination_keyword = "stop"
non_termination_keyword = "continue"
termination_function = KernelFunctionFromPrompt(
    function_name="termination", 
    prompt=f"Examine the RESPONSES and determine whether {bootcamp_agent_name} agrees by saying 'Je suis d'accord'." +
           "For the agents to agree you need to have two messages in your RESPONSES.\n" +
           f"If the content is satisfactory, respond with a single word without explanation: {termination_keyword}. Else answer just 'non'.\n" +
           "If specific suggestions are being provided, it is not satisfactory." +
           "If no correction is suggested, it is satisfactory.\n\n" +
           "RESPONSES:\n" +
           "{{$lastmessage}}"
)
# réduction de l'historque pour la fonction de terminaison
history_reducer = ChatHistoryTruncationReducer(target_count=2)
################################################

async def main():

    ################################################
    # streamlit interface, pas important
    st.title("Intro Semantic Kernel - Multi-agent chat")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=avatars.get(message["role"], None)):
            st.markdown(message["content"])
    ################################################

    # Creation du kernel
    kernel = Kernel()

    # déclaration du service LLM
    # les infos sont dans le .env
    chat_completion = AzureChatCompletion(service_id="chat-completion") #.env environment is parsed automatically

    # ajout du service au kernel
    kernel.add_service(chat_completion)
    
    # Creation des agents (qui utilisent le kernel et ont leur propre instructions)
    bootcamp_agent = ChatCompletionAgent(kernel=kernel, name=bootcamp_agent_name, instructions=BOOTCAMP_AGENT_INSTRUCTIONS)
    presenter_agent = ChatCompletionAgent(kernel=kernel, name=present_agent_name, instructions=PRESENTER_AGENT_INSTRUCTIONS)


    # Définition de la "chat room" pour agent
    chat = AgentGroupChat(agents=[bootcamp_agent, presenter_agent], # les agents
                          termination_strategy=KernelFunctionTerminationStrategy(agents=[bootcamp_agent], # comment ça se termine cette histoire ?
                                                                                 function=termination_function,
                                                                                 kernel=kernel,
                                                                                 result_parser=lambda result: termination_keyword in str(result.value[0]).lower(),
                                                                                 history_variable_name="lastmessage",
                                                                                 maximum_iterations=10,
                                                                                 history_reducer=history_reducer),
                          )

    # Lancement de la boucle de chat (l'hisorique est gérré par le kernel)
    async for response in chat.invoke():
        # affichage du message (response.content) avec le role (response.name)
        with st.chat_message(response.name, avatar=avatars.get(response.name, None)):
            st.markdown(response.content)
            st.session_state.messages.append({"role": response.name, "content": response.content})
        await asyncio.sleep(3.0)


if __name__ == "__main__":
    asyncio.run(main())