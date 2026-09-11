from constants import SCHEMA_VERSION, SOFTWARE_NAME, SOFTWARE_VERSION


def test_release_identity_uses_oct_static_software_v61_with_schema_600():
    assert SOFTWARE_NAME == "OCT_Static_Software"
    assert SOFTWARE_VERSION == "OCT_Static_Software V6.1"
    assert SCHEMA_VERSION == "6.0.0"
