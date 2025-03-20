import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

from semantic_kernel.contents.chat_history import ChatHistory

import streamlit as st

################################################
# Tracing dans le fichier telemetry : configuration des logs et connexion App Insights
from telemetry import set_up_logging, set_up_metrics, set_up_tracing


set_up_logging()
set_up_tracing()
set_up_metrics()
################################################

################################################
# Configuration de la génération du LLM
chat_settings = AzureChatPromptExecutionSettings(
    temperature=0.2,
    max_completion_tokens=100
    # service_id
    # ai_model_id
    # frequence_penalty
    # top_p
    # response_format (json ou texte)
)
################################################


################################################
# Fonction main (async)
async def main():
    # Creation du kernel
    # kernel = Kernel()

    # Instantiation du service LLM
    # Peut etre:
    # AzureOpenAI, OpenAI, Azure AI Inference (modèles déployés dans Azure)
    # Anthropic (Claude), Amazone Bedrock, Google AI, Mistral AI ....
    # Connexion information are in .env file
    # Voir .env.example
    # instruction role => system prompt
    chat_completion = AzureChatCompletion(service_id="chat-completion",
                                          instruction_role="Tu es un super assistant, aide l'utilisateur au mieux.") #.env environment is parsed automatically

    # service ajouté au kernel non obligatoire
    # kernel.add_service(chat_completion)

    # objet pour l'historique
    history = ChatHistory()


    ################################################
    # streamlit interface, pas important
    st.title("Semantic Kernel - Sans kernel, utilisation d'un LLM")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    ################################################

    ################################################
    # Récupération et affichage du prompt utilisateur
    # Prompt de l'utilisateur dans la variable prompt
    if prompt := st.chat_input("Poser une question ici..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        ################################################

        # Ajout du prompt utilisateur à l'historique
        # history.add_message({"role": "user", "content": prompt})
        history.add_user_message(prompt)

        # génération de la réponse par le LLM
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=chat_settings
            # kernel=kernel
        )

        # ajout de la réponse du llm à l'historique
        history.add_message(result)

        ################################################
        # Affichage de la réponse du LLM
        with st.chat_message("assistant"):
            st.markdown(result.content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result.content})
        ################################################



if __name__ == "__main__":
    asyncio.run(main())

    #logger = logging.getLogger("kernel")
    #logger.setLevel(loggin.DEBUG)
    #LOGGER.debug("Hello")