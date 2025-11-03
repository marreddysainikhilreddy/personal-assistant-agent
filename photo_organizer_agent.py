import os
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_agent
from typing import List

from pathlib import Path
import langchain

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from PIL import Image
import base64
from io import BytesIO


current_directory = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(current_directory, "photos")

INPUT_DIR = "./photos"
OUTPUT_DIR = "./organized"


file_toolkit = FileManagementToolkit(
    root_dir='.',
    # selected_tools=["list_directory", "move_file"]
)
file_tools = file_toolkit.get_tools()
# print(file_tools)

# For classification
llm_vision = ChatOllama(model="llama3.2-vision", temperature=0.2)

# For agent
# llm_agent = ChatOllama(model="llama3.2", temperature=0.2)
llm_agent = ChatOllama(model="llama3-groq-tool-use", temperature=0.2)


def encode_images_to_base64(image_path: str) -> str:
    with Image.open(image_path) as pil_image:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


@tool
def list_images(dir_path: str) -> List[str]:
    """ List all image files in the given directory path """
    extensions = {"jpg", "png", "jpeg", "webp"}
    path = Path(dir_path)
    images = []
    for extension in extensions:
        images.extend(path.glob(f"*.{extension}"))
        images.extend(path.glob(f"*.{extension.upper()}"))
    
    return [str(p) for p in sorted(images)]


@tool
def classify_image(image_path: str) -> str:
    """ Classify the content of the image at the given path into a category. """
    if not os.path.exists(image_path):
        return "Error: Image Not Found"
    
    base64_image = encode_images_to_base64(image_path)

    messages = [
        SystemMessage("""
            You are a photo classification assistant.
            Rules:
            - Respond ONLY with one short lowercase category (e.g. travel, family_party, pet, food... etc)
            - Reuse existing categories if they clearly fit.
            - Do NOT explain. Do NOT include punctuation or lists. Just output the single category name.
        """),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": base64_image}},
        ])
    ]
    ai_response = llm_vision.invoke(messages)
    image_category = ai_response.content.strip()
    return image_category


# This function creates a folder if a folder doesn't exist
@tool
def create_directory(dir_path: str) -> str:
    """ Create a directory at the given path if it does not exist. """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return f"Directory ready: {path}"



# System prompt for the agent
agent_system_prompt = """
    You are a smart image organizer agent.
    Goal: Move all the images from {input_dir} into subfolders of {output_base}, grouped by their content category

    How to do it:
    1. Call list_images({input_dir}) to get the full list of image paths
    2. For each image path make an appropriate tool call:
        I can see that u are mentioning moving images but use tools to do that. if u are not able to use tools then mention that.
        a. call classify_image(image_path) to get the image category
        b. compute category_folder = {output_base}/category 
        c. call create_directory(category_folder) to ensure it exists
        d. call move_file with input_path=image_path, output_path=category_folder/image_name (keep the file name same as the original)
    3. Repeat until ALL images are moved. use list_directory({input_dir}) to check if empty (ignore non-images)
    4. Keep tracking the progress and mention how many are left at each stage
    5. When there are no more images left in {input_dir}. Then respond "All images are organized successfully"
    Make tool calls until the goal is achieved

    Be precise with the paths. DO NOT STOP until the Goal of organizing images is achieved.
""".format(input_dir=INPUT_DIR, output_base=OUTPUT_DIR)

# prompt = ChatPromptTemplate.from_messages([
#     ('system', agent_system_prompt),
#     MessagesPlaceholder(variable_name="chat_history", optional=True),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

# prompt = prompt.partial(
#     input_dir=INPUT_DIR,
#     output_base=OUTPUT_DIR
# )

tools = file_tools + [list_images, classify_image, create_directory]

agent = create_agent(
    model=llm_agent,
    tools=tools,
    system_prompt=agent_system_prompt,
    debug=True
)

langchain.debug = True


result = agent.invoke({
    "messages": [{"role": "user", "content": "Start organizing ALL images now. List them first."}]
})

# print(result)

# result = agent.invoke({
#     "messages": [{"role": "user", "content": "Start organizing ALL images now. List them first."}]
# })

# print(result)

# known_categories = {}

# for filename in os.listdir(IMAGE_FOLDER):
#     if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         continue
#     # print("file name: ", filename)
#     image_path = os.path.join(IMAGE_FOLDER, filename)
#     base64_image = encode_images_to_base64(image_path)
#     # print("encode image")

#     if known_categories:
#         known_text = "\n".join([f"{k}: {v}" for k, v in known_categories.items()])
#         context_text = f"Here are the categories assigned so far:\n{known_text}\n"
#     else:
#         context_text = "No categories assigned yet. \n"
    
#     print(context_text)

#     messages = [
#         SystemMessage("""
#             You are a photo classification assistant.
#             Rules:
#             - Respond ONLY with one short lowercase category (e.g. travel, family_party, pet, food... etc)
#             - Reuse existing categories if they clearly fit.
#             - Do NOT explain. Do NOT include punctuation or lists. Just output the single category name.
#     """),
#     HumanMessage(content=[
#             {"type": "text", "text": context_text + " Classify the next image:"},
#             {"type": "image_url", "image_url": {"url": base64_image}},
#         ])
#     ]

#     ai_response = llm_vision.invoke(messages)
#     image_category = ai_response.content.strip()

#     print("Category detected: ", image_category)
    
#     known_categories[filename] = image_category




# print(known_categories)




# print(os.listdir(IMAGE_FOLDER))


## loop through the images and then for each image pass that image to the LLM and get a single category back

## After this use that category to create a folder using langchain and move that image inside that folder.

# ai_response = llama_model.invoke(messages)
# image_category = ai_response.content
# messages.append(
#     AIMessage(image_category)
# )
