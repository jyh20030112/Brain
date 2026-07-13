from simbrain.documents import loaders
from simbrain.models import DocumentPage


def test_mineru_artifact_is_written_under_project_directory(monkeypatch, tmp_path):
    source_root = tmp_path / "docs"
    source_root.mkdir()
    pdf = source_root / "guide.pdf"
    pdf.write_bytes(b"fake")
    project_dir = tmp_path / "output" / "my-knowledge-base"
    captured = {}

    monkeypatch.setattr(loaders, "_mineru_available", lambda token: (True, ""))

    def fake_mineru(path, token, output_dir):
        captured["output_dir"] = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "mineru_result.md").write_text("# 配置指南", encoding="utf-8")
        return [DocumentPage(page_number=1, text="# 配置指南")]

    monkeypatch.setattr(loaders, "_load_pdf_mineru", fake_mineru)
    docs = loaders.load_docs(
        [pdf],
        mineru_api_token="token",
        output_dir=project_dir,
        source_root=source_root,
    )

    assert captured["output_dir"].is_relative_to(project_dir / "mineru")
    assert docs[0].source_path == "guide.pdf"
    assert docs[0].metadata["parser"] == "mineru"
    assert docs[0].metadata["mineru_artifact"].startswith("mineru/")
