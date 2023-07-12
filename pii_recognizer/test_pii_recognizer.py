import os
import pytest
import random
from faker import Faker
from pii_recognizer import (
    process,
    analyzer_engine,
    anonymize,
    annotate,
    get_supported_entities,
)


def generate_routing_number():
    prefix = random.randint(0, 99)
    identifier = random.randint(0, 9999999)
    identifier_str = str(identifier).zfill(7)
    weighted_sum = (
        3 * (int(str(prefix).zfill(2)[0]))
        + 7 * (int(str(prefix).zfill(2)[1]))
        + 1 * (int(identifier_str[0]))
        + 3 * (int(identifier_str[1]))
        + 7 * (int(identifier_str[2]))
        + 1 * (int(identifier_str[3]))
        + 3 * (int(identifier_str[4]))
        + 7 * (int(identifier_str[5]))
        + 1 * (int(identifier_str[6]))
    )
    check_digit = (10 - (weighted_sum % 10)) % 10

    routing_number = f"{prefix:02d}{identifier_str}{check_digit}"

    return routing_number


def generate_us_itin():
    area_number = random.randint(900, 999)
    group_number = random.randint(70, 99)
    serial_number = random.randint(0, 9999)

    formatted_itin = f"{area_number:03d}-{group_number:02d}-{serial_number:04d}"
    return formatted_itin


@pytest.fixture(scope="function")
def fake_data(request):
    params = request.param if hasattr(request, "param") else {}
    fake = Faker("en_US")
    data = {
        "name": fake.name(),
        "email": fake.email(),
        "address": fake.address(),
        "phone": fake.phone_number(),
        "ssn": fake.ssn(),
        "credit_card": fake.credit_card_number(),
        "organization": fake.company(),
        "location": fake.street_address(),
        "date_time": fake.date(),
        "mac_address": fake.mac_address(),
        "us_bank_number": fake.bban(),
        "imei": "".join(str(fake.random_int(0, 9)) for _ in range(14)),
        "title": fake.job(),
        "license_plate": fake.license_plate(),
        "us_passport": fake.passport_number(),
        "currency": fake.currency_code(),
        "routing_number": generate_routing_number(),
        "us_itin": generate_us_itin(),
        "age": fake.random_int(1, 100),
        "password": fake.password(),
        "swift_code": fake.swift(),
    }

    data.update(params)

    yield data


def test_pattern_process(fake_data):
    ENTITIES = {
        "CREDIT_CARD": "credit_card",
        "SSN": "ssn",
        "PHONE": "phone",
        "EMAIL": "email",
    }

    analyzer = analyzer_engine("partern")
    text = f"He can be reached at {fake_data['email']} or {fake_data['phone']}.His credit card number is {fake_data['credit_card']} and his SSN is {fake_data['ssn']}."
    res, html, rpt = process(text, analyzer)

    assert any(entity in res for entity in ENTITIES.keys())


def test_spacy_process(fake_data):
    ENTITIES = {
        "PERSON": "name",
        "ORGANIZATION": "organization",
    }

    analyzer = analyzer_engine("spacy")
    text = f"{fake_data['name']}'s employer is {fake_data['organization']}."
    res, html, rpt = process(text, analyzer)

    assert any(entity in res for entity in ENTITIES.keys())


def test_flair_process(fake_data):
    ENTITIES = {
        "LOCATION": "location",
        "PERSON": "name",
        "ORGANIZATION": "organization",
        "MAC_ADDRESS": "mac_address",
        "US_BANK_NUMBER": "us_bank_number",
        "IMEI": "imei",
        "TITLE": "title",
        "LICENSE_PLATE": "license_plate",
        "US_PASSPORT": "us_passport",
        "CURRENCY": "currency",
        "ROUTING_NUMBER": "routing_number",
        "US_ITIN": "us_itin",
        "US_BANK_NUMBER": "us_bank_number",
        "AGE": "age",
        "PASSWORD": "password",
        "SWIFT_CODE": "swift_code",
    }

    analyzer = analyzer_engine("flair")
    text = " ".join(
        [item + " is " + str(fake_data[item]) for item in ENTITIES.values()]
    )
    res, html, rpt = process(text, analyzer)
    assert any(entity in res for entity in ENTITIES.keys())


def test_whole_process(fake_data):
    ENTITIES = {
        "LOCATION": "location",
        "PERSON": "name",
        "ORGANIZATION": "organization",
        "MAC_ADDRESS": "mac_address",
        "US_BANK_NUMBER": "us_bank_number",
        "IMEI": "imei",
        "TITLE": "title",
        "LICENSE_PLATE": "license_plate",
        "US_PASSPORT": "us_passport",
        "CURRENCY": "currency",
        "ROUTING_NUMBER": "routing_number",
        "US_ITIN": "us_itin",
        "US_BANK_NUMBER": "us_bank_number",
        "AGE": "age",
        "CREDIT_CARD": "credit_card",
        "SSN": "ssn",
        "PHONE": "phone",
        "EMAIL": "email",
        "PASSWORD": "password",
        "SWIFT_CODE": "swift_code",
    }
    text = " ".join(
        [item + " is " + str(fake_data[item]) for item in ENTITIES.values()]
    )
    analyzer = analyzer_engine("whole")
    res, html, rpt = process(text, analyzer)
    assert any(entity in res for entity in ENTITIES.keys())
