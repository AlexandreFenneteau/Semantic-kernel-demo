import asyncio
import sys
#import dotenv
#import logging

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.utils.logging import setup_logging

#dotenv.load_dotenv()

from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

import streamlit as st
import pandas as pd
import numpy as np

################################################
# déclaration du plugin (des points x,y)
class PointsPlugin:
    #def __init__(self, points, plot):
    def __init__(self, points):
        self.points = points
        self.plot = None

    def scatter_plot(self):
        self.plot = st.scatter_chart(st.session_state.points.points, x="x", y="y")

    @kernel_function(
        name="get_points",
        description="Gets a list of points with their coordinates",
    )
    def get_points(
        self,
    ) -> str:
        """Gets a list of points with their coordinates"""
        return self.points

    @kernel_function(
        name="add_point",
        description="Add a point to the list",
    )
    def add_point(
        self,
        x: float,
        y: float,
    ) -> str:
        """"Add a point to the list"""
        new_point = {"x": x, "y": y}
        self.points.append(new_point)
        self.plot.add_rows(pd.DataFrame([new_point]))
        return self.points
################################################



################################################
# Telemétrie
from telemetry import set_up_logging, set_up_metrics, set_up_tracing


set_up_logging()
set_up_tracing()
set_up_metrics()
################################################


async def main():
    ################################################
    # streamlit interface, pas important
    st.title("Semantic Kernel - Plugins")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    ################################################   # ajout des variables qui ne changent pas dans la session (évite de détruire lors de la reconstruction de la page streamlit)


    if "kernel" not in st.session_state:
        st.session_state.kernel = Kernel() # déclaration du kernel
        st.session_state.chat_completion=AzureChatCompletion(service_id="chat-completion") # déclaration du service LLM
        st.session_state.kernel.add_service(st.session_state.chat_completion) # ajout du service au kernel
        st.session_state.points = PointsPlugin([{"x": 0., "y": 1.}, {"x": 50., "y": -3.}])  # Déclaration du plugin, sa mémoire en session
        st.session_state.kernel.add_plugin(st.session_state.points, # ajout du pluggin au kernel
                                           plugin_name="points")
        st.session_state.history = ChatHistory() # Déclaration de l'historique de chat dans la session

    
    st.session_state.points.scatter_plot()
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    ################################################


    execution_settings = AzureChatPromptExecutionSettings()
    # Ajout de la possibilité au llm de choisir la fonction à exécuter
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    ################################################
    # Récupération et affichage du prompt utilisateur
    # Prompt de l'utilisateur dans la variable prompt
    if prompt := st.chat_input("Intéragir avec le graph ici..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        ################################################

        # Ajout du prompt utilisateur à l'historique
        # history.add_message({"role": "user", "content": prompt})
        st.session_state.history.add_user_message(prompt)

        # génération de la réponse par le LLM
        result = await st.session_state.chat_completion.get_chat_message_content(
            chat_history=st.session_state.history,
            settings=execution_settings,
            kernel=st.session_state.kernel #besoin du kernel pour intéragir avec les plugins
        )

        # ajout de la réponse du llm à l'historique
        st.session_state.history.add_message(result)

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
