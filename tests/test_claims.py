import pytest

from oracle.claims import PROHIBITED_CLAIMS, asserts_claim, normalize, unsupported_claims


def test_fifteen_approved_claims():
    assert len(PROHIBITED_CLAIMS) == 15 == len(set(PROHIBITED_CLAIMS))


@pytest.mark.parametrize("text,claim,expected", [
    ("FDA approved", "FDA approved", True),
    ("An FDA-approved formula.", "FDA approved", True),
    ("fda   APPROVED!", "FDA approved", True),
    ("not FDA approved", "FDA approved", False),
    ("This is not FDA approved", "FDA approved", False),
    ("never been FDA approved", "FDA approved", False),
    ("it isn't FDA approved", "FDA approved", False),
    ("it isn’t FDA approved", "FDA approved", False),  # curly apostrophe
    ("sold without FDA approved labels", "FDA approved", False),
    ("not really at all FDA approved", "FDA approved", True),  # negation outside the window
    ("not FDA approved, but FDA approved elsewhere", "FDA approved", True),  # one clean occurrence
    ("No side effects reported", "no side effects", True),  # "no" is part of the phrase
    ("100% original certified", "100% original certified", True),
    ("100 original certified", "100% original certified", False),
    ("money back guarantee", "money-back guarantee", True),
    ("FDA approvedness", "FDA approved", False),  # whole words only
    ("", "FDA approved", False),
])
def test_asserts_claim(text, claim, expected):
    assert asserts_claim(text, claim) is expected


def test_normalize():
    assert normalize("Isn't FDA-approved.") == ["isnt", "fda", "approved"]


def test_unsupported_claims_respect_source():
    text = "Clinically proven. Lifetime warranty."
    assert unsupported_claims(text, "") == ["clinically proven", "lifetime warranty"]
    assert unsupported_claims(text, "Comes with a lifetime warranty.") == ["clinically proven"]
