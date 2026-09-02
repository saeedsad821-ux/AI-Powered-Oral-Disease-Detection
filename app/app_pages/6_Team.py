from __future__ import annotations

import streamlit as st

from app.components.cards import render_team_card
from app.components.footer import render_footer
from app.components.header import render_page_header
from app.components.section import render_section_header

render_page_header(
    title="Team",
    subtitle="Contributors behind the oral disease detection system.",
    icon="groups",
    badges=[("5 contributors", "blue")],
)

render_section_header(
    "Contributors",
    icon="groups",
    description="Roles and focus areas for each team member.",
)

team = [
    ("Maged Awadalla Yacoub", "Project lead", "Focused on project coordination and presentation delivery.", "rocket_launch"),
    ("Ahmad Khaled Alfky", "Model & deployment", "Contributed to the deployment workflow and app structure.", "deployed_code"),
    ("Mostafa Elsayed Abd Elazeez", "Research & analysis", "Supported dataset exploration and methodology refinement.", "manage_search"),
    ("Saeed Saad Abdo Saeed", "Engineering", "Contributed to training, integration, and project polish.", "engineering"),
    ("Islam Ramadan Abdel Dayem", "Documentation & UI", "Focused on documentation quality and presentation experience.", "design_services"),
]

cols = st.columns(3)
for index, (name, role, description, icon) in enumerate(team):
    with cols[index % 3]:
        render_team_card(name, role, description, icon=icon)

render_footer()
