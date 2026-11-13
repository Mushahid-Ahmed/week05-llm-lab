import pytest

from llm_lab.schemas import ClaimAnalysis, parse_analysis

VALID = """
{
  "category": "property",
  "summary": "A pipe burst damaged the kitchen ceiling.",
  "missing_information": ["repair estimate", "damage extent"],
  "needs_human_review": true,
  "rationale": "The claim concerns household water damage."
}
"""


def test_valid_output_is_accepted() -> None:
    analysis = parse_analysis(VALID)

    assert analysis.category == "property"
    assert analysis.needs_human_review is True


def test_category_outside_allowed_list_is_rejected() -> None:
    raw = VALID.replace('"property"', '"fire"')

    with pytest.raises(ValueError, match="category"):
        parse_analysis(raw)


def test_missing_field_is_rejected() -> None:
    raw = VALID.replace('  "needs_human_review": true,\n', "")

    with pytest.raises(ValueError, match="needs_human_review"):
        parse_analysis(raw)


def test_blank_summary_is_rejected() -> None:
    raw = VALID.replace(
        "A pipe burst damaged the kitchen ceiling.",
        "   ",
    )

    with pytest.raises(ValueError, match="summary"):
        parse_analysis(raw)


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_analysis("The category is property.")


def test_missing_information_must_be_a_list_of_strings() -> None:
    raw = VALID.replace(
        '["repair estimate", "damage extent"]',
        '"repair estimate"',
    )

    with pytest.raises(ValueError, match="missing_information"):
        parse_analysis(raw)


def test_schema_forbids_extra_fields() -> None:
    with pytest.raises(ValueError, match="extra_forbidden"):
        ClaimAnalysis.model_validate_json(
            VALID.replace(
                '"rationale"',
                '"approved": true,\n  "rationale"',
            )
        )
