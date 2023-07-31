# Copyright 2019 Iguazio
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

import warnings
import os
import logging
import mlrun
import pathlib
import tempfile
import nltk
from tqdm.auto import tqdm
from typing import List, Tuple, Set, Optional, Dict, Any, Union
from collections.abc import Iterable
import presidio_analyzer as pa
import presidio_anonymizer as pre_anoymizer
from presidio_anonymizer.entities import OperatorConfig
import annotated_text.util as at_util

try:
    import flair as fl
except ModuleNotFoundError:
    print("Flair is not installed")

# There is a conflict between Rust-based tokenizers' parallel processing
# and Python's fork operations during multiprocessing. To avoid this, we need
# the following two lines

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logger = logging.getLogger("pii-recognizer")


class PatternRecognizerFactory:
    """
    Factory for creating pattern recognizers, it can be extended in the future to
    add more regex pattern for different entities. For the pattern recognizer to work,
    we need construct a list of regex patterns for each entity.
    """

    _DEFAULT_ENTITIES = {
        "CREDIT_CARD": [pa.Pattern("CREDIT_CARD", r"\b(?:\d[ -]*?){13,16}\b", 0.5)],
        "SSN": [pa.Pattern("SSN", r"\b\d{3}-?\d{2}-?\d{4}\b", 0.5)],
        "PHONE": [pa.Pattern("PHONE", r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", 0.5)],
        "EMAIL": [pa.Pattern("EMAIL", r"\S+@\S+", 0.5)],
    }

    # create a list of pattern recognizers
    @classmethod
    def _create_pattern_recognizer(cls):
        """
        For each entity, create a list of patterns to recognize it

        :param cls: PatternRecognizerFactory class

        :returns: List of pattern recognizers
        """

        # Entities to recognize and their regex patterns

        return [
            pa.PatternRecognizer(supported_entity=entity, patterns=pattern)
            for entity, pattern in cls._DEFAULT_ENTITIES.items()
        ]


class CustomSpacyRecognizer(pa.LocalRecognizer):
    """
    Custom Spacy Recognizer from Presidio Analyzer trained on Privy data.
    The privy data is generated using this https://github.com/pixie-io/pixie/tree/main/src/datagen/pii/privy
    It can be used to recognize custom entities, Since we want to use Presidio's Registries to generate AnalyzerEngine,
    it inherits from Presidio Analyzer's LocalRecognizer class.
    """

    # Entities to recognize

    _DEFAULT_ENTITIES = [
        "LOCATION",
        "PERSON",
        "NRP",
        "ORGANIZATION",
        "DATE_TIME",
    ]

    # Default explanation for this recognizer

    _DEFAULT_EXPLANATION = (
        "Identified as {} by Spacy's Named Entity Recognition (Privy-trained)"
    )

    # Label groups to check

    _DEFAULT_CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION", "STREET_ADDRESS", "COORDINATE"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"NRP"}, {"NORP", "NRP"}),
        ({"ORGANIZATION"}, {"ORG"}),
        ({"DATE_TIME"}, {"DATE_TIME"}),
    ]

    # pretrained model for this recognizer

    _DEFAULT_MODEL_LANGUAGES = {
        "en": "beki/en_spacy_pii_distilbert",
    }

    _DEFAULT_PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "NROP": "NRP",
        "DATE_TIME": "DATE_TIME",
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: List[str] = None,
        check_label_groups: Tuple[Set, Set] = None,
        context: List[str] = None,
        ner_strength: float = 1,
    ):
        """
        Initialize Spacy Recognizer.

        :param supported_language: Language to use, default is English
        :param supported_entities: Entities to use for recognition
        :param check_label_groups: Label groups to check for the entities
        :param context:            Context to use if any
        :param ner_strength:       Default confidence for NER prediction

        :returns: SpacyRecognizer object
        """

        # Default confidence for NER prediction
        self.ner_strength = ner_strength

        self.check_label_groups = check_label_groups or self._DEFAULT_CHECK_LABEL_GROUPS
        supported_entities = supported_entities or self._DEFAULT_ENTITIES
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
        )

    # get the presidio explanation for the result

    def _build_spacy_explanation(
        self, original_score: float, explanation: str
    ) -> pa.AnalysisExplanation:
        """
        Create explanation for why this result was detected.

        :param original_score: Score given by this recognizer
        :param explanation:    Explanation string

        :returns: Presidio AnalysisExplanation object
        """
        explanation = pa.AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    # main method for the recognizer
    def analyze(self, text: str, entities: List[str], nlp_artifacts=None):  # noqa D102
        """
        Analyze text using Spacy.

        :param text:          Text to analyze
        :param entities:      Entities to analyze
        :param nlp_artifacts: NLP artifacts to use

        :returns: List of Presidio RecognizerResult objects
        """
        results = []
        if not nlp_artifacts:
            logger.warning("Skipping SpaCy, nlp artifacts not provided...")
            return results

        ner_entities = nlp_artifacts.entities

        # recognize the supported entities
        for entity in entities:
            if entity not in self.supported_entities:
                continue
            for ent in ner_entities:
                if not self.__check_label(entity, ent.label_, self.check_label_groups):
                    continue

                # string of the explanation saying the entity is recognized by spacy
                textual_explanation = self._DEFAULT_EXPLANATION.format(ent.label_)
                explanation = self._build_spacy_explanation(
                    self.ner_strength, textual_explanation
                )

                # create the standard result with the entity, start, end, score, and explanation
                spacy_result = pa.RecognizerResult(
                    entity_type=entity,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=self.ner_strength,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        pa.RecognizerResult.RECOGNIZER_NAME_KEY: self.name
                    },
                )
                results.append(spacy_result)

        return results

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        """
        Check if the label is in the label group.

        :param entity:             Entity to check
        :param label:              Label to check
        :param check_label_groups: Label groups to check

        :returns: True if the label is in the label group, False otherwise
        """
        return any(
            entity in egrp and label in lgrp for egrp, lgrp in check_label_groups
        )


# Class to use Flair with Presidio as an external recognizer.
class FlairRecognizer(pa.EntityRecognizer):
    """
    Wrapper for a flair model, if needed to be used within Presidio Analyzer.
    This is to make sure the recognizer can be registered with Presidio registry.
    """

    _DEFAULT_ENTITIES = [
        "LOCATION",
        "PERSON",
        "NRP",
        "GPE",
        "ORGANIZATION",
        "MAC_ADDRESS",
        "US_BANK_NUMBER",
        "IMEI",
        "TITLE",
        "LICENSE_PLATE",
        "US_PASSPORT",
        "CURRENCY",
        "ROUTING_NUMBER",
        "US_ITIN",
        "US_BANK_NUMBER",
        "US_DRIVER_LICENSE",
        "AGE",
        "PASSWORD",
        "SWIFT_CODE",
    ]

    # This is used to construct the explanation for the result

    _DEFAULT_EXPLANATION = "Identified as {} by Flair's Named Entity Recognition"

    _DEFAULT_CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION", "STREET_ADDRESS", "COORDINATE"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"NRP"}, {"NORP", "NRP"}),
        ({"GPE"}, {"GPE"}),
        ({"ORGANIZATION"}, {"ORG"}),
        ({"MAC_ADDRESS"}, {"MAC_ADDRESS"}),
        ({"US_BANK_NUMBER"}, {"US_BANK_NUMBER"}),
        ({"IMEI"}, {"IMEI"}),
        ({"TITLE"}, {"TITLE"}),
        ({"LICENSE_PLATE"}, {"LICENSE_PLATE"}),
        ({"US_PASSPORT"}, {"US_PASSPORT"}),
        ({"CURRENCY"}, {"CURRENCY"}),
        ({"ROUTING_NUMBER"}, {"ROUTING_NUMBER"}),
        ({"AGE"}, {"AGE"}),
        ({"CURRENCY"}, {"CURRENCY"}),
        ({"SWIFT_CODE"}, {"SWIFT_CODE"}),
        ({"US_ITIN"}, {"US_ITIN"}),
        ({"US_BANK_NUMBER"}, {"US_BANK_NUMBER"}),
        ({"US_DRIVER_LICENSE"}, {"US_DRIVER_LICENSE"}),
    ]

    _DEFAULT_MODEL_LANGUAGES = {
        "en": "beki/flair-pii-distilbert",
    }

    _DEFAULT_PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "NROP": "NRP",
        "URL": "URL",
        "US_ITIN": "US_ITIN",
        "US_PASSPORT": "US_PASSPORT",
        "IBAN_CODE": "IBAN_CODE",
        "IP_ADDRESS": "IP_ADDRESS",
        "EMAIL_ADDRESS": "EMAIL",
        "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
        "US_BANK_NUMBER": "US_BANK_NUMBER",
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: List[str] = None,
        check_label_groups: Tuple[Set, Set] = None,
    ):
        """
        Initialize the FlairRecognizer.

        :param supported_language: Language to use
        :param supported_entities: Entities to use
        :param check_label_groups: Label groups to check

        :returns: FlairRecognizer object

        """
        self.check_label_groups = check_label_groups or self._DEFAULT_CHECK_LABEL_GROUPS

        supported_entities = supported_entities or self._DEFAULT_ENTITIES
        self.model = fl.models.SequenceTagger.load(
            self._DEFAULT_MODEL_LANGUAGES.get(supported_language)
        )

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="Flair Analytics",
        )

    # main method for the recognizer
    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: pa.nlp_engine.NlpArtifacts = None,
    ) -> List[pa.RecognizerResult]:
        """
        Analyze text and return the results.

        :param text:          The text for analysis.
        :param entities:      The list of entities to recognize.
        :param nlp_artifacts: Not used by this recognizer but needed for the interface.
        :param language:      Text language. Supported languages in MODEL_LANGUAGES

        :returns: The list of Presidio RecognizerResult constructed from the recognized Flair detections.
        """

        results = []

        sentences = fl.data.Sentence(text)
        self.model.predict(sentences)

        # If there are no specific list of entities, we will look for all of it.
        if not entities:
            entities = self.supported_entities

        # Go over the entities and check if they are in the supported entities list.
        for entity in entities:
            if entity not in self.supported_entities:
                continue

            # Go over the sentences and check if the entity is in the sentence.
            for ent in sentences.get_spans("ner"):
                if not self.__check_label(
                    entity, ent.labels[0].value, self.check_label_groups
                ):
                    continue

                # If the entity is in the sentence, we will add it to the results.
                textual_explanation = self._DEFAULT_EXPLANATION.format(
                    ent.labels[0].value
                )

                # Build the explanation for the result
                explanation = self._build_flair_explanation(
                    round(ent.score, 2), textual_explanation
                )

                flair_result = self._convert_to_recognizer_result(ent, explanation)

                results.append(flair_result)

        return results

    def _convert_to_recognizer_result(
        self, entity: fl.data.Span, explanation: str
    ) -> pa.RecognizerResult:
        """
        Convert Flair result to Presidio RecognizerResult.

        :param entity:      Flair entity of Span
        :param explanation: Presidio AnalysisExplanation

        :returns: Presidio RecognizerResult
        """

        # Convert the entity type to Presidio entity type
        entity_type = self._DEFAULT_PRESIDIO_EQUIVALENCES.get(entity.tag, entity.tag)

        # Convert the score to Presidio score
        flair_score = round(entity.score, 2)

        # Create the Presidio RecognizerResult from the Flair entity
        flair_results = pa.RecognizerResult(
            entity_type=entity_type,
            start=entity.start_position,
            end=entity.end_position,
            score=flair_score,
            analysis_explanation=explanation,
        )

        return flair_results

    def _build_flair_explanation(
        self, original_score: float, explanation: str
    ) -> pa.AnalysisExplanation:
        """
        Create explanation for why this result was detected.

        :param original_score: Score given by this recognizer
        :param explanation:    Explanation string

        :returns: Presidio AnalysisExplanation
        """

        # Create the Presidio AnalysisExplanation for the result
        explanation = pa.AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    # sanity check of the entity and label before recognition
    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            entity in egrp and label in lgrp for egrp, lgrp in check_label_groups
        )


# get the analyzer engine based on the model
def _get_analyzer_engine(
    score_threshold: float, model: str = "whole"
) -> pa.AnalyzerEngine:
    """
    Return pa.AnalyzerEngine.

    :param model: The model to use. Can be "spacy", "flair", "pattern" or "whole".

    :returns: pa.AnalyzerEngine
    """
    # recognizer registry that can store multiple recognizers
    registry = pa.RecognizerRegistry()

    if model == "spacy":
        # custom spacy recognizer
        spacy_recognizer = CustomSpacyRecognizer()
        # add the custom build spacy recognizer
        registry.add_recognizer(spacy_recognizer)
    if model == "flair":
        # pre-trained flair recognizer
        flair_recognizer = FlairRecognizer()
        # add the custom build flair recognizer
        registry.add_recognizer(flair_recognizer)
    if model == "pattern":
        # add the pattern recognizer
        pattern_recognizer_factory = PatternRecognizerFactory()
        for recognizer in pattern_recognizer_factory._create_pattern_recognizer():
            registry.add_recognizer(recognizer)
    if model == "whole":
        spacy_recognizer = CustomSpacyRecognizer()
        flair_recognizer = FlairRecognizer()
        registry.add_recognizer(spacy_recognizer)
        registry.add_recognizer(flair_recognizer)
        # add the pattern recognizer
        pattern_recognizer_factory = PatternRecognizerFactory()
        for recognizer in pattern_recognizer_factory._create_pattern_recognizer():
            registry.add_recognizer(recognizer)

    analyzer = pa.AnalyzerEngine(
        default_score_threshold=score_threshold,
        registry=registry,
        supported_languages=["en"],
    )
    return analyzer


def _get_anonymizer_engine() -> pre_anoymizer.AnonymizerEngine:
    """
    Return AnonymizerEngine.

    :returns: The AnonymizerEngine.
    """
    return pre_anoymizer.AnonymizerEngine()


def _anonymize(
    text: str,
    analyze_results: List[pa.RecognizerResult],
    entity_operator_map: dict = None,
    is_full_text: bool = True,
) -> str:
    """
    Anonymize identified input using Presidio Abonymizer.

    :param text:            The text for analysis.
    :param analyze_results: The list of Presidio RecognizerResult constructed from
    :param entity_operator_map: The entity_operator_map is a dictionary that maps entity to operator name and operator params.
    :param is_full_text:    Whether the text is full text or not.

    :returns: The anonymized text.
    """
    if not text:
        return ""

    anonymizer_engine = _get_anonymizer_engine()
    if not entity_operator_map:
        operators = None
    else:
        # Create OperatorConfig based on the entity_operator_map
        operators = {
            entity: OperatorConfig(operator_name, operator_params)
            for entity, (operator_name, operator_params) in entity_operator_map.items()
        }

    if is_full_text:
        # Anonymize the entire text
        return anonymizer_engine.anonymize(
            text=text, analyzer_results=analyze_results, operators=operators
        ).text
    # Tokenize the text to sentences
    sentences = nltk.sent_tokenize(text)
    anonymized_sentences = []
    current_idx = 0

    for sentence in sentences:
        start_idx = current_idx
        end_idx = start_idx + len(sentence)
        sentence_results = [
            pa.RecognizerResult(
                result.entity_type,
                start=result.start - start_idx,
                end=result.end - start_idx,
                score=result.score,
            )
            for result in analyze_results
            if result.start >= start_idx and result.end <= end_idx
        ]

        if sentence_results:  # If PII is detected
            anonymized_sentence = anonymizer_engine.anonymize(
                text=sentence, analyzer_results=sentence_results, operators=operators
            ).text
            anonymized_sentences.append(anonymized_sentence)

        current_idx = end_idx

    return " ".join(anonymized_sentences)


def _get_tokens(
    text: str, analyze_results: List[pa.RecognizerResult], is_full: bool = True
) -> List[str]:
    """
    Get the full tokens or only contains the entities that can form a sentence.

    :param text:            The text for analysis.
    :param analyze_results: The list of Presidio RecognizerResult constructed from
    :param is_full:         Whether return full tokens or just the tokens that only contains the entities that can form a sentence.

    :returns: The tokens.
    """

    tokens = []
    # sort by start index
    results = sorted(analyze_results, key=lambda x: x.start)
    for i, res in enumerate(results):
        if i == 0:
            tokens.append(text[: res.start])

        # append entity text and entity type
        tokens.append((text[res.start : res.end], res.entity_type))

        # if another entity coming i.e. we're not at the last results element,
        # add text up to next entity
        if i != len(results) - 1:
            tokens.append(text[res.end : results[i + 1].start])
        # if no more entities coming, add all remaining text
        else:
            tokens.append(text[res.end :])

    # get the tokens that only contains the entities that can form a sentence
    part_annontated_tokens = []
    if not is_full:
        last_end_sentence = 0
        for i, token in enumerate(tokens):
            if any(item in token for item in [".", "!", "?"]) and any(
                type(item) is tuple for item in tokens[last_end_sentence:i]
            ):
                part_annontated_tokens.append(tokens[last_end_sentence:i])
                last_end_sentence = i
        return part_annontated_tokens
    return tokens


def _annotate(
    text: str, st_analyze_results: List[pa.RecognizerResult], is_full_html: bool = True
) -> List[str]:
    """
    Annotate identified input using Presidio Anonymizer.

    :param text:               The text for analysis.
    :param st_analyze_results: The list of Presidio RecognizerResult constructed from analysis.
    :param is_full_html:       Whether generate full html or not.
    :returns: The list of tokens with the identified entities.

    """
    return _get_tokens(text, st_analyze_results, is_full_html)


def _process(
    text: str,
    model: pa.AnalyzerEngine,
    score_threshold: float,
    entities: List[str] = None,
    entities_operator_map: dict = None,
    is_full_text: bool = True,
) -> Tuple[str, str, str]:
    """
    Process the text of str using the model.

    :param txt:                   Text to process
    :param model:                 Model to use for processing
    :param entities:              Entities to recognize
    :param entities_operator_map: The entity_operator_map is a dictionary that maps entity to operator name and operator params.
    :param score_threshold:       The score threshold to use for recognition
    :param is_full_text:          Whether to return the full text or just the annotated text

    :returns: A tuple of:

              * the anonymized text
              * the list of Presidio RecognizerResult constructed from analysis
    """

    # get the analyzer engine

    analyzer = model

    # analyze the text that can be used for anonymization
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=entities,
        score_threshold=score_threshold,
        return_decision_process=True,
    )

    # anonymize the text, replace the pii entities with the labels
    anonymized_text = _anonymize(text, results, entities_operator_map, is_full_text)

    return anonymized_text, results


def _get_single_html(
    text: str, results: List[pa.RecognizerResult], is_full_html: bool = True
):
    """
    Generate the html for a single txt file.

    :param text: The text for analysis.
    :param results: The list of Presidio RecognizerResult constructed from analysis.
    :param is_full_html: Whether generate full html or not.

    :returns: The html string for a single txt file.
    """
    # convert the results to tokens to generate the html
    tokens = _annotate(text, results, is_full_html)
    html = at_util.get_annotated_html(*tokens)

    # avoid the error during rendering of the \n in the html
    backslash_char = "\\"

    html_str = f"<p>{html.replace('{backslash_char}n', '<br>')}</p>"

    return html_str


def _get_single_json(results: List[pa.RecognizerResult], is_full_report: bool = True):
    """
    Generate the json for a single txt file.

    :param results: The list of Presidio RecognizerResult constructed from analysis.
    :param is_full_report: Whether generate full json or not.

    :returns: The json string for a single txt file.
    """
    # generate the stats report if needed
    if not is_full_report:
        stats = []
        # add the simplify stats logic here
        for item in results:
            item.analysis_explanation = None
            stats.append(item)
    else:
        stats = results

    return stats


def _get_all_html(
    txt_content: dict,
    res_dict: dict,
    is_full_html: bool = True,
):
    """
    Generate the html for all txt files.

    :param txt_content: The dictionary of txt file name and content.
    :param res_dict:    The dictionary of txt file name and the list of Presidio RecognizerResult constructed from analysis.
    :param is_full_html: Whether generate full html or not.

    :returns: The html string for all txt files.

    """
    # These are placeholder for the html string
    html_index = "<html><head><title>Highlighted Pii Entities</title></head><body><h1>Highlighted Pii Entities</h1><ul>"
    html_content = ""
    for txt_file, results in res_dict.items():
        txt = txt_content[txt_file]
        html_index += f"<li><a href='#{txt_file}'>{txt_file}</a></li>"
        html_content += f"<li><h2>{txt_file}</h2><p>{_get_single_html(txt, results, is_full_html)}</p></li>"
    html_index += "</ul>"
    html_res = f"{html_index}{html_content}</body></html>"

    return html_res


def _get_all_rpt(res_dict: dict, is_full_report: bool = True):
    """
    Generate the stats report for all txt files.

    :param res_dict: The dictionary of txt file name and the list of Presidio RecognizerResult constructed from analysis.
    :param is_full_report: Whether generate full report or not.

    :returns: The stats report for all txt files.
    """
    # These are placeholder for the html string
    stats_dict = {}
    for txt_file, results in res_dict.items():
        new_stats = []
        for item in _get_single_json(results, is_full_report):
            if is_full_report:
                item.analysis_explanation = item.analysis_explanation.to_dict()
                new_stats.append(item.to_dict())
            else:
                tmp_dict = item.to_dict()
                tmp_dict.pop("analysis_explanation")
                tmp_dict.pop("recognition_metadata")
                new_stats.append(tmp_dict)
        stats_dict[txt_file] = new_stats
    return stats_dict


def recognize_pii(
    context: mlrun.MLClientCtx,
    input_path: str,
    output_path: str,
    output_suffix: str,
    html_key: str,
    score_threshold: float,
    entities: List[
        str
    ] = None,  # List of entities to recognize, default is recognize all
    entity_operator_map: dict = None,
    model: str = "whole",
    generate_json: bool = True,
    generate_html: bool = True,
    is_full_text: bool = True,
    is_full_html: bool = True,
    is_full_report: bool = True,
) -> Tuple[pathlib.Path, dict, dict]:
    """
    Walk through the input path, recognize PII in text and store the anonymized text in the output path. Generate the html with different colors for each entity, json report of the explaination.

    :param context:              The MLRun context. this is needed for log the artifacts.
    :param entities:             The list of entities to recognize.
    :param entity_operator_map:  The map of entity to operator (mask, redact, replace, keep, hash, and its params)
    :param input_path:           The input path of the text files needs to be analyzied.
    :param output_path:          The output path to store the anonymized text.
    :param output_suffix:        The surfix of output key for the anonymized text. for example if the input file is pii.txt, the output key is anoymized, the output file name will be pii_anonymized.txt.
    :param html_key:             The html key for the artifact.
    :param score_threshold:      The score threshold to mark the recognition as trusted.
    :param model:                The model to use. Can be "spacy", "flair", "pattern" or "whole".
    :param is_generate_json_rpt: Whether to generate the json report of the explaination.
    :param is_generate_html:     Whether to generate the html report of the explaination.
    :param is_full_text:         Whether to return the full text or only the masked text.
    :param is_full_html:         Whether to return the full html or just the annotated text
    :param is_full_report:       Whether to return the full report or just the score and start, end index

    :returns: A tuple of:

              * Path to the output directory
              * The json report of the explaination (if is_generate_json_rpt is True)
              * A dictionary of errors files that were not processed

    """

    # Set output directory
    if output_path is None:
        output_path = tempfile.mkdtemp()

    # Create the output directory:
    output_directory = pathlib.Path(output_path)
    if not output_directory.exists():
        output_directory.mkdir()

    # Load the model:
    analyzer = _get_analyzer_engine(model)
    print("Model loaded")

    # Go over the text files in the input path, analyze and anonymize them:
    txt_files_directory = pathlib.Path(input_path)
    errors = {}

    res_dict = {}
    txt_content = {}
    # Go over the text files in the input path, analyze and anonymize them:
    for i, txt_file in enumerate(
        tqdm(
            list(txt_files_directory.glob("*.txt")),
            desc="Processing files",
            unit="file",
        )
    ):
        try:
            # Load the str from the text file
            text = txt_file.read_text()
            txt_content[str(txt_file)] = text
            # Process the text to recoginze the pii entities in it
            anonymized_text, results = _process(
                text,
                analyzer,
                entities,
                entity_operator_map,
                score_threshold,
                is_full_text,
            )
            res_dict[str(txt_file)] = results
            # Store the anonymized text in the output path
            output_file = (
                output_directory
                / f"{str(txt_file.relative_to(txt_files_directory)).split('.')[0]}_{output_suffix}.txt"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(anonymized_text)

        except Exception as e:
            errors[str(txt_file)] = str(e)
            print(f"Error processing {txt_file}: {e}")
    if generate_html:
        # Generate the html report
        html_res = _get_all_html(txt_content, res_dict, is_full_html)
        # Store the html report in the context
        arti_html = mlrun.artifacts.Artifact(body=html_res, format="html", key=html_key)
        context.log_artifact(arti_html)
    if generate_json:
        # Generate the json report
        json_res = _get_all_rpt(res_dict, is_full_report)
        return output_path, json_res, errors
    return output_path, errors
