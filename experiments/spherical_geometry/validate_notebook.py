"""CI smoke test only: execute the notebook using explicitly synthetic input."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import nbformat
from nbclient import NotebookClient
from PIL import Image
from build_standard_gt import build_standard_gt

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    images = root/"synthetic_images"
    images.mkdir()
    for i, x in enumerate([0, 1, 3]):
        Image.new("RGB", (32, 16)).save(images/f"equi_{i}_{x}_0_0_0_0_0.png")
    sequence = build_standard_gt(images, root/"gt", sequence_id="synthetic_not_research_data")
    notebook = nbformat.read(ROOT/"notebooks/10_spherical_geometry_phase1.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.startswith("# Experiment settings"):
            cell.source = (
                f"CONFIG = Config(sequence={str(sequence)!r}, output={str(root/'output')!r}, "
                "baseline_edges_m=(0.,2.,4.), pairs_per_bin=2, grid_height=5, grid_width=9, "
                "static_formats=(), image_diagnostics=True)\n"
                "print('CI synthetic fixture; not research measurements')"
            )
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    assert (root/"output/run_status.json").is_file()
    print("Notebook executed successfully with synthetic input. No research results generated.")
