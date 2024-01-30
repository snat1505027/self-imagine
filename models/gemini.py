import os
import google.generativeai as genai
from pathlib import Path
from PIL import Image


class Gemini:
    def __init__(self, name='gemini-pro-vision', **kwargs) -> None:
        genai.configure(api_key=os.environ['GENAI_API_KEY'])
        self.model = genai.GenerativeModel(name)

    def ask(self, img_path, text:str, **kwargs):
        response = self.model.generate_content([text, Image.open(img_path)])
        return response.text
