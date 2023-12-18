# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mlrun
import os
import tqdm
from langchain.chat_models import ChatOpenAI
import ast
from src.common import ProjectSecrets


def generate_data(
        context: mlrun.MLClientCtx,
        fields: list,
        amount: int = 10,
        model_name: str = "gpt-3.5-turbo",
        language: str = "en",
        chunk_size: int = 50,
) -> list:
    """
    Generates a Json file with data randomly made from given user fields.

    :param context: mlrun context.
    :param fields: A list of fields to randomly generate.
    :param amount: The number of variants to generate.
    :param model_name: The name of the model to use for conversation generation.
                       You should choose one of GPT-4 or GPT-3.5 from the list here: https://platform.openai.com/docs/models.
                       Default: 'gpt-3.5-turbo'.
    :param language: The language to use for the generated conversation text.
    :param chunk_size: Number of samples generated at each GPT query.
    """
    instructions = ""
    for field in fields:
        if ":" in field:
            key, instruction = field.split(":", 1)
        else:
            key, instruction = field, "no special instruction"
        key = key.replace(" ", "_")
        instructions += f"* {key}: {instruction}\n"

    # Create the prompt structure:
    prompt_structure = (
        f"Use the following keys and instructions (example: 'key: instruction or no special instruction'): {instructions}.\n"
        f"Please generate the values in {language} language. \n"
        f"Make sure the names of the keys are the same as the given field name.\n"
        f"Please return only the json format without any introduction and ending"
    )

    prompt_structure = "generate the following values {amount} times randomly, in an order that creates a json table.\n" + prompt_structure

    # Load the OpenAI model using langchain:
    os.environ["OPENAI_API_KEY"] = context.get_secret(key=ProjectSecrets.OPENAI_API_KEY)
    os.environ["OPENAI_API_BASE"] = context.get_secret(
        key=ProjectSecrets.OPENAI_API_BASE
    )
    llm = ChatOpenAI(model=model_name)

    # Start generating data:
    data = []
    for _ in tqdm.tqdm(range(int(amount / chunk_size) + 1), desc="Generating"):
        for tryout in range(3):
            if amount > chunk_size:
                current_chunk_size = chunk_size
                amount -= chunk_size
            else:
                current_chunk_size = amount

            # Create the prompt:
            prompt = prompt_structure.format(
                amount=current_chunk_size,
            )
            # Generate the conversation:
            chunk_data = llm.predict(text=prompt)
            chunk_data = chunk_data[chunk_data.find("["):chunk_data.rfind("]") + 1]
            if chunk_data.count("[") != chunk_data.count("]"):
                print("Failed to get proper json format from model, number of '[' doesn't match number of ']'.")
                continue
            chunk_data = ast.literal_eval(chunk_data)
            data += chunk_data
            break
        if tryout == 3:
            raise ValueError(
                f"Could not generate a proper json format for the given fields, using given model: {model_name}. Hint: Gpt-4 works for our purpose.")
    return data
