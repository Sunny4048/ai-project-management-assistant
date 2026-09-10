import streamlit as st
from huggingface_hub import InferenceClient


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Project Management Assistant",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Application title and introduction
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Function selection
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Input area
# ---------------------------------------------------------

project_input = st.text_area(
    "Enter your project information:",
    height=220,
    placeholder="Enter the relevant project information here..."
)


# ---------------------------------------------------------
# Qwen AI function
# ---------------------------------------------------------

def ask_qwen(system_prompt, user_prompt):

    client = InferenceClient(
        api_key=st.secrets["HF_TOKEN"],
        provider="featherless-ai"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B-Instruct",
        messages=messages,
        max_tokens=800,
        temperature=0.2
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Project Plan
# ---------------------------------------------------------

def generate_project_plan(project_information):

    system_prompt = (
        "You are an IT project management assistant. "
        "Create a practical and structured project plan "
        "using the information provided by the user. "
        "Do not present missing information as an established "
        "project fact. Do not invent specific budgets, dates, "
        "deadlines, requirements, staffing problems, legal "
        "issues, technical problems, or other project facts "
        "unless they are provided by the user. "
        "When information is unavailable, write "
        "'Not provided' or 'Requires confirmation'."
    )

    user_prompt = (
        "Create a structured project plan from the following "
        "information.\n\n"
        "Include, where appropriate:\n"
        "- Project overview\n"
        "- Objectives\n"
        "- Scope\n"
        "- Deliverables\n"
        "- Project team\n"
        "- Stakeholders\n"
        "- Activities\n"
        "- Communication plan\n"
        "- Quality assurance\n"
        "- Budget information\n"
        "- Potential assumptions or information gaps\n\n"
        "Clearly identify information that was not provided.\n\n"
        "Project information:\n"
        + project_information
    )

    return ask_qwen(system_prompt, user_prompt)


# ---------------------------------------------------------
# Risk Register
# ---------------------------------------------------------

def generate_risk_register(project_information):

    system_prompt = (
        "You are an IT project management assistant. "
        "Create a practical risk register based on the "
        "information provided by the user. "
        "Do not present missing or unknown information as "
        "an established fact about the project. "
        "You may identify reasonable potential risks, but "
        "clearly describe them as potential risks rather "
        "than confirmed project conditions. "
        "Do not invent project-specific facts such as actual "
        "funding problems, legal issues, staffing problems, "
        "technical problems, deadlines, or requirements "
        "unless they are stated by the user. "
        "When information is unavailable, state "
        "'Not provided' or 'Requires confirmation'."
    )

    user_prompt = (
        "Create a structured risk register for the following "
        "project information.\n\n"
        "For each risk, include:\n"
        "- Risk ID\n"
        "- Risk description\n"
        "- Likelihood\n"
        "- Impact\n"
        "- Risk level\n"
        "- Mitigation/action\n\n"
        "Clearly distinguish between risks supported by the "
        "provided information and potential risks that "
        "require confirmation.\n\n"
        "Do not claim that unprovided problems already exist.\n\n"
        "Project information:\n"
        + project_information
    )

    return ask_qwen(system_prompt, user_prompt)


# ---------------------------------------------------------
# Meeting Minutes
# ---------------------------------------------------------

def generate_meeting_minutes(meeting_information):

    system_prompt = (
        "You are an IT project management assistant. "
        "Create professional and structured meeting minutes "
        "using only the information provided by the user. "
        "Do not invent attendees, dates, decisions, actions, "
        "deadlines, discussion points, or responsibilities. "
        "If information is missing, clearly state "
        "'Not provided' or 'Requires confirmation'."
    )

    user_prompt = (
        "Create professional meeting minutes from the "
        "following information.\n\n"
        "Include:\n"
        "- Meeting purpose\n"
        "- Date and time, if provided\n"
        "- Attendees\n"
        "- Key discussion points\n"
        "- Decisions\n"
        "- Action items\n"
        "- Responsible persons\n"
        "- Deadlines\n"
        "- Information requiring confirmation\n\n"
        "Do not invent information that was not provided.\n\n"
        "Meeting information:\n"
        + meeting_information
    )

    return ask_qwen(system_prompt, user_prompt)


# ---------------------------------------------------------
# Project Summary
# ---------------------------------------------------------

def generate_project_summary(project_information):

    system_prompt = (
        "You are an IT project management assistant. "
        "Create a concise and accurate project summary using "
        "only the information provided by the user. "
        "Do not invent project facts. "
        "Do not assume that missing activities, deadlines, "
        "budgets, risks, achievements, or project status "
        "actually exist. "
        "If information is missing, state "
        "'Not provided' or 'Requires confirmation'."
    )

    user_prompt = (
        "Create a professional project summary from the "
        "following information.\n\n"
        "Include:\n"
        "- Project overview\n"
        "- Main objectives\n"
        "- Scope\n"
        "- Key stakeholders\n"
        "- Project team\n"
        "- Current activities or status, if provided\n"
        "- Key deliverables\n"
        "- Risks or issues mentioned by the user\n"
        "- Important information gaps\n\n"
        "Clearly distinguish between provided information "
        "and information that requires confirmation.\n\n"
        "Project information:\n"
        + project_information
    )

    return ask_qwen(system_prompt, user_prompt)


# ---------------------------------------------------------
# Stakeholder Communication
# ---------------------------------------------------------

def generate_stakeholder_communication(project_information):

    system_prompt = (
        "You are an IT project management assistant. "
        "Create professional stakeholder communication using "
        "only the information provided by the user. "
        "Do not invent project progress, achievements, "
        "deadlines, problems, budgets, decisions, or other "
        "project facts. "
        "If important information is missing, clearly mark it "
        "as 'Not provided' or 'Requires confirmation'."
    )

    user_prompt = (
        "Prepare a professional stakeholder communication "
        "message based on the following project information.\n\n"
        "Include, where supported by the information:\n"
        "- Subject\n"
        "- Project status or overview\n"
        "- Key activities\n"
        "- Important updates\n"
        "- Risks or issues\n"
        "- Required stakeholder actions\n"
        "- Next steps\n\n"
        "Do not invent information. "
        "Clearly identify information that requires confirmation.\n\n"
        "Project information:\n"
        + project_information
    )

    return ask_qwen(system_prompt, user_prompt)


# ---------------------------------------------------------
# Generate Output button
# ---------------------------------------------------------

if st.button("Generate Output"):

    # Empty input protection
    if not project_input.strip():

        st.error(
            "Please enter some project information before "
            "generating an output."
        )

    # Long input protection
    elif len(project_input) > 10000:

        st.error(
            "Your input is too long. Please shorten the "
            "project information and try again."
        )

    # Project Plan
    elif function == "Project Plan":

        with st.spinner("Generating project plan..."):

            try:

                output = generate_project_plan(project_input)

                st.subheader("Generated Project Plan")

                st.markdown(output)

            except Exception:

                st.error(
                    "The AI service could not generate the "
                    "requested output at this time. "
                    "Please try again later."
                )

    # Risk Register
    elif function == "Risk Register":

        with st.spinner("Generating risk register..."):

            try:

                output = generate_risk_register(project_input)

                st.subheader("Generated Risk Register")

                st.markdown(output)

            except Exception:

                st.error(
                    "The AI service could not generate the "
                    "requested output at this time. "
                    "Please try again later."
                )

    # Meeting Minutes
    elif function == "Meeting Minutes":

        with st.spinner("Generating meeting minutes..."):

            try:

                output = generate_meeting_minutes(project_input)

                st.subheader("Generated Meeting Minutes")

                st.markdown(output)

            except Exception:

                st.error(
                    "The AI service could not generate the "
                    "requested output at this time. "
                    "Please try again later."
                )

    # Project Summary
    elif function == "Project Summary":

        with st.spinner("Generating project summary..."):

            try:

                output = generate_project_summary(project_input)

                st.subheader("Generated Project Summary")

                st.markdown(output)

            except Exception:

                st.error(
                    "The AI service could not generate the "
                    "requested output at this time. "
                    "Please try again later."
                )

    # Stakeholder Communication
    elif function == "Stakeholder Communication":

        with st.spinner(
            "Generating stakeholder communication..."
        ):

            try:

                output = generate_stakeholder_communication(
                    project_input
                )

                st.subheader(
                    "Generated Stakeholder Communication"
                )

                st.markdown(output)

            except Exception:

                st.error(
                    "The AI service could not generate the "
                    "requested output at this time. "
                    "Please try again later."
                )
