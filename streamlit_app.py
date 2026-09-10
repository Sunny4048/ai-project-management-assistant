import streamlit as st
from huggingface_hub import InferenceClient

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


def generate_project_plan(project_information):

    client = InferenceClient(
        api_key=st.secrets["HF_TOKEN"],
        provider="featherless-ai"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an IT project management assistant. "
                "Generate a practical project plan using only "
                "the information provided by the user. "
                "Do not invent unsupported project facts. "
                "If information is missing, clearly identify "
                "it as an assumption or information gap."
            )
        },
        {
            "role": "user",
            "content": (
                "Create a structured project plan from the "
                "following project information:\n\n"
                + project_information
            )
        }
    ]

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=messages,
        max_tokens=700,
        temperature=0.2
    )

    return response.choices[0].message.content
    def generate_risk_register(project_information):

    client = InferenceClient(
        api_key=st.secrets["HF_TOKEN"],
        provider="featherless-ai"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an IT project management assistant. "
                "Create a practical risk register using only "
                "the information provided by the user. "
                "Do not invent unsupported project facts. "
                "If information is missing, clearly identify "
                "it as an assumption or information gap."
            )
        },
        {
            "role": "user",
            "content": (
                "Create a structured risk register for the "
                "following project information. For each risk, "
                "include the risk description, likelihood, "
                "impact, risk level, and mitigation/action.\n\n"
                + project_information
            )
        }
    ]

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=messages,
        max_tokens=700,
        temperature=0.2
    )

    return response.choices[0].message.content


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

    elif function != "Project Plan":

        st.info(
            "This function will be connected in the next step. "
            "Project Plan is currently being tested."
        )

    else:

        with st.spinner("Generating project plan..."):

            try:

                output = generate_project_plan(project_input)

                st.subheader("Generated Project Plan")

                st.write(output)

            except Exception:

                st.error(
                    "The AI service could not generate an output "
                    "at this time. Please check the configuration "
                    "and try again."
                )
                
