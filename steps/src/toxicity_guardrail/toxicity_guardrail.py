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


class ToxicityGuardrailStep:
    """
    A serving graph step that filters out toxic requests using a pre-trained
    text classification model.

    If the toxicity score of the input text meets or exceeds the threshold,
    the request is blocked with a ValueError. Safe requests are passed through
    unchanged.

    The classifier label "toxic" maps directly to the toxicity score; any
    other label (e.g. "non-toxic") inverts the model's confidence score.
    """

    def __init__(
        self,
        context=None,
        name=None,
        threshold: float = 0.5,
        model_name: str = "unitary/toxic-bert",
        **kwargs,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self._classifier = None

    def post_init(self, mode="sync", **kwargs):
        from transformers import pipeline

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
