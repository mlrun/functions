# Copyright 2025 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Dict
from transformers import pipeline

class ToxicityGuardrailStep:
    def __init__(
        self,
        context=None,
        name=None,
        threshold: float = 0.5,
        model_name: str = "unitary/toxic-bert",
        **kwargs,
    ):
        """
        A serving graph step that filters out toxic requests using a pre-trained
        text classification model.

        :param context: MLRun context object, injected automatically by the serving graph.
        :param name: Name of this step in the serving graph.
        :param threshold: Toxicity score threshold; requests whose toxicity score meets or
                          exceeds this value are blocked with a ValueError. Defaults to 0.5.
        :param model_name: HuggingFace model identifier used for text classification.
                           Defaults to "unitary/toxic-bert".
        :param kwargs: Additional keyword arguments forwarded to the serving graph step base.
        """
        self.threshold = threshold
        self.model_name = model_name
        self._classifier = None

    def post_init(self, mode="sync", **kwargs):
        self._classifier = pipeline("text-classification", model=self.model_name)

    def do(self, event: Dict[str, Any]) -> Dict[str, Any]:
        question = event.get("question", "")
        result = self._classifier(question)[0]
        score = (
            result["score"]
            if result["label"] == "toxic"
            else 1 - result["score"]
        )
        if score >= self.threshold:
            raise ValueError(
                f"Request blocked: toxicity score {score:.3f} >= {self.threshold}"
            )
        return event
