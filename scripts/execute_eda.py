"""Execute notebooks/01_EDA.ipynb end-to-end in the DataAnalytics kernel."""

import asyncio
import sys
import warnings

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

warnings.filterwarnings("ignore")

nb = nbformat.read("notebooks/01_EDA.ipynb", as_version=4)
nbformat.validate(nb)
print("1. nbformat validation: PASS")

client = NotebookClient(nb, timeout=1200, kernel_name="dataanalytics",
                        resources={"metadata": {"path": "."}})
client.execute()
print("2. Full notebook execution: SUCCESS")

code_cells = [c for c in nb.cells if c["cell_type"] == "code"]
with_out = [c for c in code_cells if c.get("outputs")]
imgs = sum(1 for c in code_cells for o in c.get("outputs", [])
           if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}))
errs = sum(1 for c in code_cells for o in c.get("outputs", [])
           if o.get("output_type") == "error")
print(f"3. cells with outputs: {len(with_out)}/{len(code_cells)}, "
      f"embedded charts: {imgs}, errors: {errs}")

nbformat.write(nb, "notebooks/01_EDA.ipynb")
print("4. Outputs written back to 01_EDA.ipynb")
