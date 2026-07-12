import pytest

from brain.project import ProjectLock, ProjectLockedError, atomic_write_json, read_json, validate_project_name


@pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b", "bad\x00name", "x" * 129])
def test_validate_project_name_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        validate_project_name(value)


def test_project_lock_rejects_concurrent_ingestion(tmp_path):
    project_dir = tmp_path / "my-knowledge-base"
    with ProjectLock(project_dir):
        with pytest.raises(ProjectLockedError):
            with ProjectLock(project_dir):
                pass


def test_atomic_write_json_replaces_complete_document(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"version": 1})
    atomic_write_json(path, {"version": 2})

    assert read_json(path) == {"version": 2}
