import streamlit as st

st.set_page_config(
    page_title="AI Project Management Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Project Management Assistant")

st.write(
    "A Generative AI-based assistant designed to support "
    "selected IT project management activities."
)

st.warning(
    "⚠️ AI-generated information may contain inaccurate or "
    "unsupported content. Please verify all outputs before "
    "using them for project-management decisions."
)

st.subheader("Select a project-management function")

function = st.selectbox(
    "Choose a function:",
    [
        "Project Plan",
        "Risk Register",
        "Meeting Minutes",
        "Project Summary",
        "Stakeholder Communication"
    ]
)

project_input = st.text_area(
    "Enter your project information:",
    height=200,
    placeholder="Enter the relevant project information here..."
)

if st.button("Generate Output"):

    if not project_input.strip():
        st.error(
            "Please enter some project information before "
            "generating an output."
        )

    elif len(project_input) > 10000:
        st.error(
            "Your input is too long. Please shorten the "
            "project information and try again."
        )

    else:
        st.success(
            f"Ready to generate a {function}."
        )

        st.write(
            "Your project information has been received."
        )
