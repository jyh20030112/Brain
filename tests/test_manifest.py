from brain.manifest import build_manifest


def test_build_manifest_merges_inventory_and_generates_description():
    manifest = build_manifest(
        project="my-knowledge-base",
        workspace_id="wid",
        embedding_model="embedding-model",
        embedding_dim=2,
        active_index="docs_wid_current",
        index_version="docs_wid_current_v_1",
        inventory=[
            {
                "file_name": "guide.pdf",
                "source_path": "guide.pdf",
                "file_type": "pdf",
                "title": "访问权限配置",
                "parser": "mineru",
                "page_count": 3,
                "chunk_count": 8,
            }
        ],
        previous_manifest=None,
        incoming_files={
            "guide.pdf": {
                "title": "访问权限配置",
                "sha256": "abc",
                "size_bytes": 123,
                "updated_at": "2026-07-12T00:00:00+00:00",
            }
        },
    )

    assert manifest["description"] == "包含 1 份资料，主题包括：访问权限配置。"
    assert manifest["topics"] == ["访问权限配置"]
    assert manifest["files"][0]["sha256"] == "abc"
    assert manifest["chunk_count"] == 8
