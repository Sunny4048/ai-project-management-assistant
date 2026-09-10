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


def build_prompt(function_name, project_information):

    base_rules = """
You are an AI assistant supporting IT project management.

Use ONLY the information provided by the user.

Do not invent project facts, dates, people, budgets, risks,
requirements, stakeholders, deadlines, or other details.

If information is missing, write "Not provided" or
"Requires confirmation" rather than making something up.

Clearly label suggestions as "Proposed" or "Potential".

Keep the answer practical, concise, and suitable for an
IT project manager.
"""

    if function_name == "Project Plan":

        return f"""
{base_rules}

Create a simple project plan based only on the project
information below.

Include:

1. Project objective
2. Key activities
3. Project phase/status
4. Roles or responsibilities mentioned by the user
5. Dependencies or items requiring confirmation

Project information:

{project_information}
"""

    elif function_name == "Risk Register":

        return f"""
{base_rules}

Create a simple risk register based only on the project
information below.

Use a table with:

- Risk
- Description
- Likelihood
- Impact
- Proposed mitigation

If likelihood, impact, or mitigation is not provided,
write "Requires confirmation" or "Proposed".

Project information:

{project_information}
"""

    elif function_name == "Meeting Minutes":

        return f"""
{base_rules}

Create structured meeting minutes from the information below.

Include:

- Meeting purpose
- Key discussion points
- Decisions
- Action items
- Owners
- Deadlines

Do not invent decisions, owners, or deadlines.
Use "Not provided" where information is missing.

Project information:

{project_information}
"""

    elif function_name == "Project Summary":

        return f"""
{base_rules}

Create a concise project summary based only on the
information below.

Include:

- Project overview
- Objective
- Current status
- Team information
- Key activities
- Important missing information requiring confirmation

Project information:

{project_information}
"""

    else:

        return f"""
{base_rules}

Draft a professional stakeholder communication based only
on the information below.

Include:

- Subject
- Greeting
- Purpose/update
- Key information
- Any action required
- Closing

Do not invent stakeholder names, dates, decisions, or
project facts.

Use "Not provided" where information is missing.

Project information:

{project_information}
"""


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

    elif "HF_TOKEN" not in st.secrets:

        st.error(
            "The Hugging Face token is not configured. "
            "Please add HF_TOKEN in the Streamlit app Secrets."
        )

    else:

        try:

            with st.spinner("Generating output..."):

                client = InferenceClient(
                    provider="featherless-ai",
                    api_key=st.secrets["HF_TOKEN"]
                )

                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-3B-Instruct",

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful IT project "
                                "management assistant. Follow the "
                                "user's instructions and never invent "
                                "unsupported project facts."
                            )
                        },
                        {
                            "role": "user",
                            "content": build_prompt(
                                function,
                                project_input
                            )
                        }
                    ],

                    max_tokens=900,
                    temperature=0.0
                )

                output = response.choices[0].message.content

                st.subheader("Generated Output")

                st.write(output)

                st.download_button(
                    label="Download Output",
                    data=output,
                    file_name="project_management_output.txt",
                    mime="text/plain"
                )

        except Exception:

            st.error(
                "The AI service could not generate the response. "
                "Please try again. If the problem continues, "
                "check the Streamlit app logs."
            )
