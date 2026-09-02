"""Execute a notebook end-to-end in the DataAnalytics kernel.

Usage: python execute_notebook.py <path-to-ipynb>
"""

import asyncio
import sys
import warnings

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

warnings.filterwarnings("ignore")

path = sys.argv[1] if len(sys.argv) > 1 else "notebooks/02_Preprocessing.ipynb"
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 7200

nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
print("1. nbformat validation: PASS")

client = NotebookClient(nb, timeout=timeout, kernel_name="dataanalytics",
                        resources={"metadata": {"path": "."}})
try:
    client.execute()
    print("2. Full notebook execution: SUCCESS")
except Exception as exc:  # nbclient CellExecutionError etc.
    print("2. Execution FAILED:", type(exc).__name__)
    for i, c in enumerate(nb.cells):
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print(f"   failing cell #{i}")
                src = c["source"]
                print("   --- code ---")
                print((src if isinstance(src, str) else "".join(src))[:600])
                print("   --- traceback (tail) ---")
                for t in o.get("traceback", [])[-12:]:
                    print("   " + t[:220].strip())
    nbformat.write(nb, path)
    print("3. Partial outputs written to", path)
    sys.exit(1)

code_cells = [c for c in nb.cells if c["cell_type"] == "code"]
with_out = [c for c in code_cells if c.get("outputs")]
imgs = sum(1 for c in code_cells for o in c.get("outputs", [])
           if o.get("output_type") == "display_data" and "image/png" in o.get("data", {}))
errs = sum(1 for c in code_cells for o in c.get("outputs", [])
           if o.get("output_type") == "error")
print(f"3. cells with outputs: {len(with_out)}/{len(code_cells)}, "
      f"embedded charts: {imgs}, errors: {errs}")

nbformat.write(nb, path)
print("4. Outputs written back to notebook")
