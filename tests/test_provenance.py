import hashlib

from experiments.provenance import FROZEN_FILES, REPO_ROOT, frozen_provenance


def test_provenance_records_each_frozen_file():
    prov = frozen_provenance()
    assert set(prov["files"]) == set(FROZEN_FILES)
    for rel, info in prov["files"].items():
        assert info["sha256"] == hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
        assert set(info) == {"last_commit", "sha256", "dirty"}
    assert isinstance(prov["any_dirty"], bool) and len(prov["head"]) == 40
