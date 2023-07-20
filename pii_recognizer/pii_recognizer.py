import mlrun
from mlrun.datastore import DataItem
from mlrun.artifacts import Artifact
from presidio_anonymizer import AnonymizerEngine
from annotated_text.util import get_annotated_html
from json import JSONEncoder
import json
import warnings
import os
import logging
from typing import Optional, List, Tuple, Set

from presidio_analyzer import (
    RecognizerRegistry,
    AnalyzerEngine,
    RecognizerResult,
    LocalRecognizer,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    AnalysisExplanation,
)
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers.spacy_recognizer import SpacyRecognizer

try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
except ModuleNotFoundError:
    print("Flair is not installed")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logger = logging.getLogger("pii-recognizer")


class PatternRecognizerFactory:
    @staticmethod
    def create_pattern_recognizer():
        ENTITIES = {
            "CREDIT_CARD": [Pattern("CREDIT_CARD", r"\b(?:\d[ -]*?){13,16}\b", 0.5)],
            "SSN": [Pattern("SSN", r"\b\d{3}-?\d{2}-?\d{4}\b", 0.5)],
            "PHONE": [Pattern("PHONE", r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", 0.5)],
            "EMAIL": [Pattern("EMAIL", r"\S+@\S+", 0.5)],
        }
        res = []
        for entity, pattern in ENTITIES.items():
            res.append(PatternRecognizer(supported_entity=entity, patterns=pattern))
        return res


class CustomSpacyRecognizer(LocalRecognizer):
    ENTITIES = [
        "LOCATION",
        "PERSON",
        "NRP",
        "ORGANIZATION",
        "DATE_TIME",
    ]

    DEFAULT_EXPLANATION = (
        "Identified as {} by Spacy's Named Entity Recognition (Privy-trained)"
    )

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION", "STREET_ADDRESS", "COORDINATE"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"NRP"}, {"NORP", "NRP"}),
        ({"ORGANIZATION"}, {"ORG"}),
        ({"DATE_TIME"}, {"DATE_TIME"}),
    ]

    MODEL_LANGUAGES = {
        "en": "beki/en_spacy_pii_distilbert",
    }

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION",
        "NROP": "NRP",
        "DATE_TIME": "DATE_TIME",
    }

    def __init__(
        self,
        supported_language: str = "en",
        supported_entities: Optional[List[str]] = None,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        context: Optional[List[str]] = None,
        ner_strength: float = 0.85,
    ):
        """
        Initialize Spacy Recognizer.
        :param supported_language: Language to use
        :param supported_entities: Entities to use
        :param check_label_groups: Label groups to check
        :param context: Context to use
        :param ner_strength: NER strength to use
        :returns SpacyRecognizer object
        """
        self.ner_strength = ner_strength
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )
        supported_entities = supported_entities if supported_entities else self.ENTITIES
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
        )

    def load(self) -> None:
        """Load the model, not used. Model is loaded during initialization."""
        pass

    def get_supported_entities(self) -> List[str]:
        """
        Return supported entities by this model.
        :returns List of the supported entities.
        """
        return self.supported_entities

    def build_spacy_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :returns: AnalysisExplanation object
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    def analyze(self, text, entities, nlp_artifacts=None):  # noqa D102
        """
        Analyze text using Spacy.
        :param text: Text to analyze
        :param entities: Entities to analyze
        :param nlp_artifacts: NLP artifacts to use
        :returns: List of RecognizerResult objects
        """
        results = []
        if not nlp_artifacts:
            logger.warning("Skipping SpaCy, nlp artifacts not provided...")
            return results

        ner_entities = nlp_artifacts.entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue
            for ent in ner_entities:
                if not self.__check_label(entity, ent.label_, self.check_label_groups):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(ent.label_)
                explanation = self.build_spacy_explanation(
                    self.ner_strength, textual_explanation
                )
                spacy_result = RecognizerResult(
                    entity_type=entity,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=self.ner_strength,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name
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
        :param entity: Entity to check
        :param label: Label to check
        :param check_label_groups: Label groups to check
        :returns: True if the label is in the label group, False otherwise
        """
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )


class FlairRecognizer(EntityRecognizer):
    """
    Wrapper for a flair model, if needed to be used within Presidio Analyzer.
    """

    ENTITIES = [
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

    DEFAULT_EXPLANATION = "Identified as {} by Flair's Named Entity Recognition"

    CHECK_LABEL_GROUPS = [
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

    MODEL_LANGUAGES = {
        "en": "beki/flair-pii-distilbert",
    }

    PRESIDIO_EQUIVALENCES = {
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
        supported_entities: Optional[List[str]] = None,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        model: SequenceTagger = None,
    ):
        """
        Initialize the FlairRecognizer.
        :param supported_language: Language to use
        :param supported_entities: Entities to use
        :param check_label_groups: Label groups to check
        :param model: Flair model to use
        :returns: FlairRecognizer object

        """
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )

        supported_entities = supported_entities if supported_entities else self.ENTITIES
        self.model = (
            model
            if model
            else SequenceTagger.load(self.MODEL_LANGUAGES.get(supported_language))
        )

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="Flair Analytics",
        )

    def load(self) -> None:
        """Load the model, not used. Model is loaded during initialization."""
        pass

    def get_supported_entities(self) -> List[str]:
        """
        Return supported entities by this model.
        :returns: List of the supported entities.
        """
        return self.supported_entities

    # Class to use Flair with Presidio as an external recognizer.
    def analyze(
        self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Analyze text using Text Analytics.
        :param text: The text for analysis.
        :param entities: Not working properly for this recognizer.
        :param nlp_artifacts: Not used by this recognizer.
        :param language: Text language. Supported languages in MODEL_LANGUAGES
        :returns: The list of Presidio RecognizerResult constructed from the recognized
            Flair detections.
        """

        results = []

        sentences = Sentence(text)
        self.model.predict(sentences)

        # If there are no specific list of entities, we will look for all of it.
        if not entities:
            entities = self.supported_entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue

            for ent in sentences.get_spans("ner"):
                if not self.__check_label(
                    entity, ent.labels[0].value, self.check_label_groups
                ):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(
                    ent.labels[0].value
                )
                explanation = self.build_flair_explanation(
                    round(ent.score, 2), textual_explanation
                )
                flair_result = self._convert_to_recognizer_result(ent, explanation)

                results.append(flair_result)

        return results

    def _convert_to_recognizer_result(self, entity, explanation) -> RecognizerResult:
        """
        Convert Flair result to Presidio RecognizerResult.
        :param entity: Flair entity
        :param explanation: AnalysisExplanation
        :returns: RecognizerResult
        """
        entity_type = self.PRESIDIO_EQUIVALENCES.get(entity.tag, entity.tag)
        flair_score = round(entity.score, 2)

        flair_results = RecognizerResult(
            entity_type=entity_type,
            start=entity.start_position,
            end=entity.end_position,
            score=flair_score,
            analysis_explanation=explanation,
        )

        return flair_results

    def build_flair_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :returns: AnalysisExplanation
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )


def analyzer_engine(model="whole"):
    """Return AnalyzerEngine.
    :param model: The model to use. Can be "spacy", "flair", "pattern" or "whole".
    :returns: AnalyzerEngine
    """
    registry = RecognizerRegistry()

    spacy_recognizer = CustomSpacyRecognizer()

    flair_recognizer = FlairRecognizer()

    if model == "spacy":
        # add the custom build spacy recognizer
        registry.add_recognizer(spacy_recognizer)
    if model == "flair":
        # add the custom build flair recognizer
        registry.add_recognizer(flair_recognizer)
    if model == "pattern":
        # add the pattern recognizer
        pattern_recognizer_factory = PatternRecognizerFactory()
        for recognizer in pattern_recognizer_factory.create_pattern_recognizer():
            registry.add_recognizer(recognizer)
    if model == "whole":
        registry.add_recognizer(spacy_recognizer)
        registry.add_recognizer(flair_recognizer)
        # add the pattern recognizer
        pattern_recognizer_factory = PatternRecognizerFactory()
        for recognizer in pattern_recognizer_factory.create_pattern_recognizer():
            registry.add_recognizer(recognizer)

    analyzer = AnalyzerEngine(registry=registry, supported_languages=["en"])
    return analyzer


def anonymizer_engine():
    """Return AnonymizerEngine.
    :returns: The AnonymizerEngine.
    """
    return AnonymizerEngine()


def get_supported_entities():
    """Return supported entities from the Analyzer Engine.
    :returns: The list of supported entities.
    """
    return analyzer_engine().get_supported_entities()


def analyze(**kwargs):
    """Analyze input using Analyzer engine and input arguments (kwargs).
    :param kwargs: The input arguments for the analyzer engine.
    :returns: The list of Presidio RecognizerResult constructed from the recognized
    """
    if "entities" not in kwargs or "All" in kwargs["entities"]:
        kwargs["entities"] = None
    return analyzer_engine().analyze(**kwargs)


def anonymize(text, analyze_results):
    """Anonymize identified input using Presidio Abonymizer.
    :param text: The text for analysis.
    :param analyze_results: The list of Presidio RecognizerResult constructed from
    :returns: The anonymized text.
    """
    if not text:
        return
    res = anonymizer_engine().anonymize(text, analyze_results)
    return res.text


def annotate(text, st_analyze_results, st_entities):
    """Annotate identified input using Presidio Anonymizer.
    :param text: The text for analysis.
    :param st_analyze_results: The list of Presidio RecognizerResult constructed from
    :returns: The list of tokens with the identified entities.

    """
    tokens = []
    # sort by start index
    results = sorted(st_analyze_results, key=lambda x: x.start)
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
    return tokens


class CustomEncoder(JSONEncoder):
    """Custom encoder for JSONEncoder.
    :param JSONEncoder: The JSONEncoder.
    :returns: The JSONEncoder.
    """

    def default(self, o):
        return o.__dict__


def process(text: str, model: AnalyzerEngine):
    """
    Process the text of str using the model.
    :param txt: Text to process
    :param model: Model to use for processing
    :returns: The anonymized text, the html string and the stats report.
    """
    analyzer = model
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=get_supported_entities(),
        return_decision_process=True,
    )
    anonymized_text = anonymize(text, results)

    annotated_tokens = annotate(text, results, get_supported_entities())
    html = get_annotated_html(*annotated_tokens)
    backslash_char = "\\"

    html_str = f"<p>{html.replace('{backslash_char}n', '<br>')}</p>"

    stats = results

    return anonymized_text, html_str, stats


def recognize_pii(
    context: mlrun.MLClientCtx,
    input_path: str,
    output_path: str,
    output_suffix: str,
    html_key: str,
    model: str = "whole",
) -> str:
    """
    Recognize PII in text.
    :param context: The MLRun context.
    :param input_path: The input path to the artifact.
    :param output_path: The output path to store the anonymized text.
    :param output_suffix: The surfix of output key for the anonymized text. for example if the input file is pii.txt, the output key will be pii_anonymized.txt.
    :param html_key: The html key for the artifact.
    :param model: The model to use. Can be "spacy", "flair", "pattern" or "whole".
    :returns: output_path

    """
    if not output_path:
        output_path = input_path + "/output/"
    analyzer = analyzer_engine(model)
    txt_file_paths, txt_file_names = get_text_files(input_path)
    html_index = "<html><head><title>Highlighted Pii Entities</title></head><body><h1>Highlighted Pii Entities</h1><ul>"
    html_content = ""
    rpt_json = {}
    for file_path, file_name in zip(txt_file_paths, txt_file_names):
        text = mlrun.get_dataitem(file_path).get().decode("utf-8")
        anonymized_text, html_str, stats = process(text, analyzer)
        html_index += f'<li><a href="#{file_name}">{file_name}</a></li>'
        html_content += f'<h2 id="{file_name}">{file_name}</h2>{html_str}'
        with open(f"{output_path}/{file_name}_{output_suffix}.txt", "w") as f:
            f.write(anonymized_text)
        new_stats = []
        for item in stats:
            item.analysis_explanation = item.analysis_explanation.to_dict()
            new_stats.append(item.to_dict())
        rpt_json[file_name] = new_stats
    html_index += "</ul>"
    html_res = f"{html_index}{html_content}</body></html>"
    arti_html = Artifact(body=html_res, format="html", key=html_key)
    context.log_artifact(arti_html)
    return rpt_json, output_path


def get_text_files(path):
    """
    Get a list of text file paths from a given path.
    :param path: The path to walk through.
    :returns: A list of text file paths and list of file names.
    """
    txt_file_lst = []
    txt_file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                txt_file_lst.append(os.path.join(root, file))
                txt_file_names.append(file[:-4])
        break
    return txt_file_lst, txt_file_names
