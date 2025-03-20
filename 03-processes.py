import asyncio
from enum import Enum
from typing import ClassVar

from pydantic import Field

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextToImage
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_context import KernelProcessStepContext
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState
from semantic_kernel.processes.local_runtime.local_event import KernelProcessEvent
from semantic_kernel.processes.local_runtime.local_kernel_process import start
from semantic_kernel.processes.process_builder import ProcessBuilder

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

import streamlit as st


SERVICE_LINKEDIN_GENERATOR = "linkedin-content-generator"
SERVICE_DALLE="dalle"

class IntroStep(KernelProcessStep):
    @kernel_function
    async def print_intro_message(self):
        print("Welcome to Processes in Semantic Kernel.\n")

class EventsIDs(Enum):
    StartProcess = "startProcess"
    PostGenerated = "PostGenerated"
    ImgPromptGenerated = "ImgPromptGenerated"
    ImageGenerated = "ImageGenerated"
    UserTopicReceived = "UserTopicReceived"
    Topic_set = "topic_set"
    Post_set = "post_set"
    Img_set = "img_set"
    Exit = "exit"

class UserInputStep(KernelProcessStep):
    GET_USER_TOPIC: ClassVar[str] = "get_user_topic"

    @kernel_function(name=GET_USER_TOPIC)
    async def get_user_topic(self, context: KernelProcessStepContext):
        """Gets the user input."""
        # topic = input("TOPIC: ")

        topic = None
        if topic := st.chat_input("Le sujet de votre article..."):
            st.chat_message("user", avatar='🤓').markdown(topic)
            await context.emit_event(process_event=EventsIDs.UserTopicReceived, data=topic)


class ArticleState(KernelBaseModel):
    """The state object for ChatBotResponseStep."""
    topic: str = ""
    linkedin_post_text: str = ""
    linkedin_img_url: str = ""


class ArticleGenerator(KernelProcessStep[ArticleState]):
    SET_TOPIC: ClassVar[str] = "set_topic"
    SET_POST: ClassVar[str] = "set_post"
    SET_IMG: ClassVar[str] = "set_img"
    # DISPLAY_POST: ClassVar[str] = "display_post"

    state: ArticleState = Field(default_factory=ArticleState)

    async def activate(self, state: "KernelProcessStepState[ArticleState]"):
        """Activates the step and initializes the state object."""
        self.state = state.state or ArticleState()
        self.state.topic = self.state.topic or ""
        self.state.linkedin_post_text = self.state.linkedin_post_text or ""
        self.state.linkedin_img_url = self.state.linkedin_img_url or ""

    @kernel_function(name=SET_TOPIC)
    async def set_topic(self, context: "KernelProcessStepContext", topic: str, kernel: "Kernel"):
        self.state.topic = topic
        print(f"Topic received: {topic}")

        await context.emit_event(process_event=EventsIDs.Topic_set, data=topic)

    @kernel_function(name=SET_POST)
    async def set_post(self, context: "KernelProcessStepContext", post: str, kernel: "Kernel"):
        self.state.linkedin_post_text = post
        print(f"Post received: {post}")

        await context.emit_event(process_event=EventsIDs.Post_set, data=post)

    @kernel_function(name=SET_IMG)
    async def set_img(self, context: "KernelProcessStepContext", img_url: str, kernel: "Kernel"):
        self.state.linkedin_img_url = img_url

        print("Img set")
        print("ARTICLE GENERATED:\n" + 
              f"{self.state.linkedin_img_url}\n" +
              f"{self.state.linkedin_post_text}")

        
        st.image(self.state.linkedin_img_url)
        st.chat_message("assistant", avatar=r"assets\cadre.jpg").markdown(self.state.linkedin_post_text)
        await context.emit_event(process_event=EventsIDs.Exit)

    # @kernel_function(name=DISPLAY_POST)
    # async def display_post(self, context: "KernelProcessStepContext", kernel: "Kernel"):
    #     print("ARTICLE GENERATED:\n" + 
    #           f"{self.state.linkedin_img_url}\n" +
    #           f"{self.state.linkedin_post_text}")

    #     await context.emit_event(process_event=EventsIDs.Exit)



class LinkedinGenerator(KernelProcessStep):
    GENERATE_LINKEDIN_TEXT: ClassVar[str] = "generate_linkedin_text_post"
    # GENERATE_DALLE_PROMPT: ClassVar[str] = "generate_dalle_prompt"

    @kernel_function(name=GENERATE_LINKEDIN_TEXT)
    async def generate_linkedin_text_post(self, context: "KernelProcessStepContext", topic: str, kernel: "Kernel"):
        """Generates a linkedin post based on the prompt."""
        # Get chat completion service and generate a response
        chat_service: AzureChatCompletion = kernel.get_service(service_id=SERVICE_LINKEDIN_GENERATOR)
        settings = AzureChatPromptExecutionSettings(service_id=SERVICE_LINKEDIN_GENERATOR, max_tokens=500, temperature=0.2)

        chat_history = ChatHistory()
        chat_history.add_system_message("Tu es un assistant qui aide l'utilisateur à rédiger des postes linkedin." + \
                                        " A chaque fois que l'utilisateur donne un sujet, génère un post linkedin inspirant sur le sujet." + \
                                        " Surtout, fais en des caisses sur le côté professionnel et utilise des mots clés très corporate.")
        chat_history.add_user_message(topic)
        response = await chat_service.get_chat_message_contents(chat_history=chat_history, settings=settings)

        if response is None:
            raise ValueError("Failed to get a response from the chat completion service.")

        answer = response[0].content

        print(f"{SERVICE_LINKEDIN_GENERATOR}: {answer}")
        st.chat_message("assistant", avatar=r"assets\linkedin.png").markdown(answer)

        # Emit an event: assistantResponse
        await context.emit_event(process_event=EventsIDs.PostGenerated, data=answer)

class DalleePromptGenerator(KernelProcessStep):
    GENERATE_DALLE_PROMPT: ClassVar[str] = "generate_dalle_prompt"

    @kernel_function(name=GENERATE_DALLE_PROMPT)
    async def generate_dalle_prompt(self, context: "KernelProcessStepContext", post: str, kernel: "Kernel"):
        """Generates a prompt for the image of linkedin post."""
        # Add user message to the state
        # Get chat completion service and generate a response
        print("Entree generate_dalle_prompt")
        chat_service: AzureChatCompletion = kernel.get_service(service_id=SERVICE_LINKEDIN_GENERATOR)
        settings = AzureChatPromptExecutionSettings(service_id=SERVICE_LINKEDIN_GENERATOR)

        chat_history = ChatHistory()
        chat_history.add_user_message("Génère moi un prompt pour générer une image avec dall-e-3 qui colle le mieux au POST LINKEDIN. Répond uniquement le prompt.\n\n" +
                                      "POST LINKEDIN:\n" + post)
        response = await chat_service.get_chat_message_contents(chat_history=chat_history, settings=settings)

        if response is None:
            raise ValueError("Failed to get a response from the chat completion service.")

        dalle_prompt = response[0].content

        print(f"DALLE PROMPT GENERATOR: {dalle_prompt}")

        st.chat_message("assistant", avatar=r"assets\prompt.jpg").markdown(dalle_prompt)
        # Emit an event: assistantResponse
        await context.emit_event(process_event=EventsIDs.ImgPromptGenerated, data=dalle_prompt)

    
class ImgGenerator(KernelProcessStep):
    GENERATE_DALLE_IMG: ClassVar[str] = "generate_dalle_img"

    @kernel_function(name=GENERATE_DALLE_IMG)
    async def generate_dalle_img(self, context: "KernelProcessStepContext", prompt: str, kernel: "Kernel"):
        """Generates an based on the prompt."""
        # Add user message to the state
        # Get chat completion service and generate a response
        dalle_service: AzureTextToImage = kernel.get_service(service_id=SERVICE_DALLE)

        img_link = await dalle_service.generate_image(prompt, 1024, 1024)

        print(f"DALLE GENERATED IMAGE: {img_link}")
        st.chat_message("assistant", avatar=r"assets\dalle.jpg").markdown(img_link)

        # Emit an event: assistantResponse
        await context.emit_event(process_event=EventsIDs.ImageGenerated, data=img_link)

kernel = Kernel()


async def step01_processes(scripted: bool = True):
    st.title("Semantic Kernel - Processes")


    kernel.add_service(AzureChatCompletion(service_id=SERVICE_LINKEDIN_GENERATOR, instruction_role="Tu es un super assistant qui aide l'utilisateur."))
    kernel.add_service(
        AzureTextToImage(service_id=SERVICE_DALLE,
                                        deployment_name="dall-e-3",
                                        api_version="2024-05-01-preview")
    )

    process = ProcessBuilder(name="ChatBot")

    # Define the steps on the process builder based on their types, not concrete objects
    intro_step = process.add_step(IntroStep)
    user_input_step = process.add_step(UserInputStep)
    article_step = process.add_step(ArticleGenerator)
    linkedin_post_step = process.add_step(LinkedinGenerator)
    dalle_prompt_step = process.add_step(DalleePromptGenerator)
    dalle_generation_step = process.add_step(ImgGenerator)
    #response_step = process.add_step(ChatBotResponseStep)

    # Define the input event that starts the process and where to send it
    process.on_input_event(event_id=EventsIDs.StartProcess).send_event_to(target=intro_step)

    # Define the event that triggers the next step in the process
    intro_step.on_function_result(function_name=IntroStep.print_intro_message.__name__).send_event_to(
        target=user_input_step, function_name=UserInputStep.GET_USER_TOPIC
    )

    # For the user step, send the user input to the response step
    user_input_step.on_event(event_id=EventsIDs.UserTopicReceived).send_event_to(
        target=article_step, parameter_name="topic", function_name=ArticleGenerator.SET_TOPIC,
    )

    article_step.on_event(event_id=EventsIDs.Topic_set).send_event_to(
        target=linkedin_post_step, parameter_name="topic", function_name=LinkedinGenerator.GENERATE_LINKEDIN_TEXT
    )

    linkedin_post_step.on_event(event_id=EventsIDs.PostGenerated).send_event_to(
        target=article_step, parameter_name="post", function_name=ArticleGenerator.SET_POST
    )

    article_step.on_event(event_id=EventsIDs.Post_set).send_event_to(
        target=dalle_prompt_step, parameter_name="post", function_name=DalleePromptGenerator.GENERATE_DALLE_PROMPT
    )

    dalle_prompt_step.on_event(event_id=EventsIDs.ImgPromptGenerated).send_event_to(
        target=dalle_generation_step, parameter_name="prompt", function_name=ImgGenerator.GENERATE_DALLE_IMG
    )

    dalle_generation_step.on_event(event_id=EventsIDs.ImageGenerated).send_event_to(
        target=article_step, parameter_name="img_url", function_name=ArticleGenerator.SET_IMG
    )

    article_step.on_event(event_id=EventsIDs.Exit).stop_process()

    # Build the kernel process
    kernel_process = process.build()

    # Start the process
    await start(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id=EventsIDs.StartProcess, data=None),
    )


if __name__ == "__main__":
    # if you want to run this sample with your won input, set the below parameter to False
    asyncio.run(step01_processes(scripted=False))