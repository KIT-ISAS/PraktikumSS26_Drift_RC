"""Unit tests for the ``Topics`` registry invariants."""

from shared_pkg.topics import Topics


def _topics() -> dict[str, str]:
    """Return all public topic-name attributes of the Topics class."""
    return {
        name: value
        for name, value in vars(Topics).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def test_registry_is_not_empty():
    assert len(_topics()) > 0


def test_all_topics_are_nonempty_absolute_names():
    for name, value in _topics().items():
        assert value, f"{name} is empty"
        assert value.startswith("/"), f"{name} = {value!r} is not an absolute topic name"
        assert not value.endswith("/"), f"{name} = {value!r} has a trailing slash"


def test_no_duplicate_topic_names():
    topics = _topics()
    values = list(topics.values())
    duplicates = {value for value in values if values.count(value) > 1}
    assert not duplicates, f"Duplicate topic names would cause silent routing bugs: {duplicates}"


def test_namespaced_topics_extend_their_base():
    topics = _topics()
    bases = {name: value for name, value in topics.items() if name.endswith("_BASE")}
    assert bases, "Expected at least one *_BASE namespace attribute"
    for base_name, base_value in bases.items():
        group_prefix = base_name.removesuffix("_BASE") + "_"
        group = {
            name: value
            for name, value in topics.items()
            if name.startswith(group_prefix) and name != base_name
        }
        for name, value in group.items():
            assert value.startswith(base_value + "/"), (
                f"{name} = {value!r} does not extend its namespace {base_name} = {base_value!r}"
            )
