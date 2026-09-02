from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
for path in (str(ROOT), str(APP_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(ROOT))
sys.path.append(str(APP_ROOT))

try:
    from app.components.sidebar import render_sidebar
except ModuleNotFoundError:
    from components.sidebar import render_sidebar

try:
    from app.components.styles import inject_styles, render_loading_splash
except ModuleNotFoundError:
    from components.styles import inject_styles, render_loading_splash

st.set_page_config(page_title="Oral Vision", page_icon=":material/medical_services:", layout="wide")

inject_styles()
render_loading_splash()

render_sidebar()

pages_dir = APP_ROOT / "app_pages"
pages = {
    "Overview": [
        st.Page(pages_dir / "1_Home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Analysis": [
        st.Page(pages_dir / "2_Disease_Detection.py", title="Disease detection", icon=":material/biotech:"),
        st.Page(pages_dir / "3_Model_Comparison.py", title="Model comparison", icon=":material/query_stats:"),
        st.Page(pages_dir / "4_Dataset_Insights.py", title="Dataset insights", icon=":material/insights:"),
    ],
    "About": [
        st.Page(pages_dir / "5_About_Project.py", title="About project", icon=":material/info:"),
        st.Page(pages_dir / "6_Team.py", title="Team", icon=":material/groups:"),
    ],
}

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
