"""Verify generated notebook syntax, source completeness and its real-data smoke."""
import ast
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / "artifacts/colab/bankruptcy_training_colab.ipynb").read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        ast.parse(cell["source"])
smoke_cell = notebook["cells"][8]["source"]
tree = ast.parse(smoke_cell)
smoke_call = tree.body[2].value
smoke = ast.literal_eval(smoke_call.args[0].elts[2])
assert "s.train_indices" in smoke and "s.test_indices" not in smoke
with zipfile.ZipFile(root / "artifacts/colab/bankruptcy_colab_source.zip") as archive:
    names = archive.namelist()
    assert not any(name.startswith("Database/") or "__pycache__" in name for name in names)
    for required in ("ruff.toml", ".github/workflows/ci.yml", "docs/RELEASE_CHECKLIST.md",
                     "tests/fixtures/feature_schema.csv", "config/training.yaml"):
        assert required in names, required
exec(compile(smoke, "colab-smoke", "exec"))
print("Notebook syntax, bundle contents and train-only smoke verified.")
