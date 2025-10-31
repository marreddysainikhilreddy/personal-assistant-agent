import os
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import base64
from io import BytesIO


current_directory = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(current_directory, "photos")
print(IMAGE_FOLDER)



toolkit = FileManagementToolkit(
    root_dir=IMAGE_FOLDER
)
# print(toolkit.get_tools())

llama_model = ChatOllama(model="llama3.2-vision", temperature=0.2)


def encode_images_to_base64(image_path: str) -> str:
    with Image.open(image_path) as pil_image:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# print(encode_images_to_base64(IMAGE_FOLDER + "/trip.jpeg"))


prompt = """
    You are organizing a set of photos into folders based on their visual content.

    For each new image:
    - Analyze it
    - Respond with exaclty one short category (lowercase, underscore if needed)
    - If this image matches any previous category you've seen in this conversation, reuse that category.
    - Do not explain, just return the category
"""
base64_image = encode_images_to_base64(IMAGE_FOLDER + "/5.jpg")

messages = [
    SystemMessage("You are a photo organization assistant. Respond only with one short lowercase category, no explanation."),
    HumanMessage(prompt)   
]



known_categories = {}

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    print("file name: ", filename)
    image_path = os.path.join(IMAGE_FOLDER, filename)
    base64_image = encode_images_to_base64(image_path)
    print("encode image")

    messages.append(
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": base64_image}},
            {"type": "text", "text": prompt}
        ])
    )

    print("Added message")
    ai_response = llama_model.invoke(messages)
    image_category = ai_response.content.strip()

    print("Category detected: ", image_category)
    
    known_categories[filename] = image_category

    # messages.append(
    #     AIMessage(f"The image {filename} belongs to the category: {image_category}" )
    # )



print(known_categories)
# print(os.listdir(IMAGE_FOLDER))






## loop through the images and then for each image pass that image to the LLM and get a single category back

## After this use that category to create a folder using langchain and move that image inside that folder.

# ai_response = llama_model.invoke(messages)
# image_category = ai_response.content
# messages.append(
#     AIMessage(image_category)
# )
