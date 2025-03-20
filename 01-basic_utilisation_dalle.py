import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings, AzureTextToImage

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

    # service ajouté au kernel non obligatoire
    # kernel.add_service(chat_completion)

    # objet pour l'historique
    # history = ChatHistory()


    ################################################
    # streamlit interface, pas important
    st.title("Semantic Kernel - Dall-E3")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.img_links = []

    # Display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        st.image(st.session_state.img_links[i])
    ################################################

    ################################################
    # Récupération et affichage du prompt utilisateur
    # Prompt de l'utilisateur dans la variable prompt
    if prompt := st.chat_input("Générer une image ici..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        ################################################


        image_generation = AzureTextToImage(service_id="text-to-image",
                                        deployment_name="dall-e-3",
                                        api_version="2024-05-01-preview")

        img_link = await image_generation.generate_image(prompt, 1024, 1024)
        st.image(img_link)
        st.session_state.img_links.append(img_link)



if __name__ == "__main__":
    asyncio.run(main())

    #logger = logging.getLogger("kernel")
    #logger.setLevel(loggin.DEBUG)
    #LOGGER.debug("Hello")