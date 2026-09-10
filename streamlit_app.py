import streamlit as st
from huggingface_hub import InferenceClient

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Project Management Assistant",
    page_icon="📊",
    layout="wide",
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_INPUT_CHARS = 10000
MAX_NEW_TOKENS = 900

FUNCTIONS = {
    "Project Plan": {
        "description": "Generate a structured draft project plan.",
        "instruction": """
Create a structured project plan using ONLY the project information supplied by
the user.

Important rules:
- Do not invent project facts.
- Do not invent names, dates, budgets, technologies, requirements, deadlines,
  stakeholders, resources or project status.
- If a required fact is missing, write "Not provided" or "Requires confirmation".
- You may propose reasonable generic tasks or planning suggestions, but clearly
  label them as "Proposed" rather than presenting them as confirmed project facts.
- Keep confirmed information and proposed information clearly separated.

Use these headings:
1. Project Overview
2. Objectives
3. Key Deliverables
4. Proposed Tasks
5. Milestones / Dates
6. Dependencies
7. Risks / Issues
8. Assumptions Requiring Confirmation
9. Next Steps
""",
    },
    "Risk Register": {
        "description": "Generate a structured draft risk register.",
        "instruction": """
Create a project risk register using ONLY the project information supplied by
the user.

Important rules:
- Do not claim that an unsupported risk is a confirmed project fact.
- Separate risks explicitly mentioned by the user from generic potential risks.
- For generic risks, label them "Potential / requires confirmation".
- Do not invent owners, dates, budgets, technologies or project-specific events.
- If likelihood, impact, owner or mitigation information is missing, write
  "Not provided" or "Requires confirmation".
- Suggested mitigations may be proposed, but label them as proposed actions.

Use a table-like structure with:
Risk | Source/Status | Likelihood | Impact | Proposed Mitigation | Owner | Confirmation Required

Then provide:
Key Risk Notes
Human Verification Required
""",
    },
    "Meeting Minutes": {
        "description": "Turn meeting information into structured minutes.",
        "instruction": """
Create structured meeting minutes using ONLY the information supplied by the
user.

Important rules:
- Do not invent attendees, decisions, action owners, deadlines or meeting dates.
- Do not infer that a discussion resulted in a decision unless the user states it.
- If information is missing, write "Not provided".
- Clearly distinguish decisions from discussion points.
- Do not create action items that were not stated or clearly requested.

Use these headings:
1. Meeting Details
2. Attendees
3. Key Discussion Points
4. Decisions
5. Action Items
6. Issues / Open Questions
7. Next Steps
8. Information Requiring Confirmation
""",
    },
    "Project Summary": {
        "description": "Generate a concise project summary.",
        "instruction": """
Create a concise project summary using ONLY the information supplied by the user.

Important rules:
- Do not invent project facts.
- Do not invent project status, dates, budget, people, technologies,
  achievements or problems.
- If information is missing, write "Not provided".
- Do not convert assumptions into facts.
- Keep the summary concise and suitable for a project-management context.

Use these headings:
1. Project Overview
2. Objectives
3. Current Status
4. Key Activities
5. Issues / Risks
6. Next Steps
7. Information Requiring Confirmation
""",
    },
    "Stakeholder Communication": {
        "description": "Generate a draft stakeholder project update.",
        "instruction": """
Draft a professional stakeholder communication using ONLY the information
supplied by the user.

Important rules:
- Do not invent stakeholder names, dates, progress, achievements, risks,
  deadlines, budgets or project status.
- If information is missing, use "Not provided" or a clear placeholder.
- Do not present assumptions as confirmed facts.
- Do not claim that work has been completed unless the user states this.
- Keep the communication concise, professional and suitable for review by a
  human project manager.

Use this structure:
Subject
Project Update
Current Status
Progress / Completed Activities
Current Issues or Risks
Next Steps
Information Requiring Confirmation
""",
    },
}


# ------------------------------------------------------------
# Hugging Face connection
# ------------------------------------------------------------
@st.cache_resource
def get_hf_client():
    """Create one cached Hugging Face InferenceClient."""
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        return None

    if not hf_token or not str(hf_token).strip():
        return None

    return InferenceClient(
        provider="auto",
        api_key=str(hf_token).strip(),
    )


def generate_response(function_name: str, project_input: str) -> str:
    """Generate a project-management response through Hugging Face."""
    client = get_hf_client()

    if client is None:
        raise RuntimeError(
            "The Hugging Face API token has not been configured. "
            "Please add HF_TOKEN in the Streamlit app Secrets settings."
        )

    function = FUNCTIONS[function_name]

    system_prompt = """You are an AI assistant supporting IT project management.

Your role is to help a human project manager draft and organise information.
You are NOT an autonomous project manager.

Accuracy rules are mandatory:
- Use only facts supplied by the user as confirmed project information.
- Never fabricate project-specific facts.
- Never invent names, dates, budgets, requirements, technologies, stakeholders,
  project status, decisions, deadlines or completed work.
- When information is missing, explicitly say "Not provided" or
  "Requires confirmation".
- If you make a generic suggestion, clearly label it as proposed or potential.
- Do not make unsupported statements sound certain.
- Produce a clear, professional, structured response.
- The human user must verify the output before using it in a real project.
"""

    user_prompt = f"""
Selected project-management function:
{function_name}

Task-specific instructions:
{function["instruction"]}

Project information supplied by the user:
--- BEGIN USER INFORMATION ---
{project_input}
--- END USER INFORMATION ---

Generate the requested output now.
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )

        if not completion or not getattr(completion, "choices", None):
            raise RuntimeError("The AI service returned an empty response.")

        message = completion.choices[0].message
        output = getattr(message, "content", None)

        if not output or not str(output).strip():
            raise RuntimeError("The AI service returned no usable text.")

        return str(output).strip()

    except Exception as exc:
        error_text = str(exc).strip()

        # Keep technical details out of the normal user interface while still
        # making the failure understandable.
        if "401" in error_text or "Unauthorized" in error_text:
            raise RuntimeError(
                "The AI service rejected the Hugging Face token. "
                "Please check the HF_TOKEN in Streamlit Secrets."
            ) from exc

        if "429" in error_text or "rate" in error_text.lower():
            raise RuntimeError(
                "The AI service is temporarily rate-limited or out of available "
                "inference capacity. Please try again later."
            ) from exc

        if "402" in error_text or "payment" in error_text.lower():
            raise RuntimeError(
                "The selected Hugging Face inference provider requires available "
                "credits for this request. Please check your Hugging Face "
                "Inference Providers account and credits."
            ) from exc

        raise RuntimeError(
            "The AI service could not generate the response. "
            "Please try again. If the problem continues, check the app logs."
        ) from exc


# ------------------------------------------------------------
# User interface
# ------------------------------------------------------------
st.title("📊 AI Project Management Assistant")

st.write(
    "A Generative AI-based assistant designed to support selected "
    "IT project-management activities."
)

st.warning(
    "⚠️ AI-generated information may contain inaccurate or unsupported "
    "content. Please verify all outputs before using them for "
    "project-management decisions."
)

st.info(
    "Human review is required. The assistant generates drafts and "
    "suggestions; it does not replace project-manager judgement."
)

st.subheader("1. Select a project-management function")

function = st.selectbox(
    "Choose a function:",
    list(FUNCTIONS.keys()),
)

st.caption(FUNCTIONS[function]["description"])

st.subheader("2. Enter project information")

project_input = st.text_area(
    "Project information",
    height=250,
    max_chars=MAX_INPUT_CHARS,
    placeholder=(
        "Enter the relevant project information here. "
        "For example: project objectives, tasks, meeting notes, "
        "known risks, current status, stakeholders or next steps."
    ),
    help=f"Maximum input length: {MAX_INPUT_CHARS:,} characters.",
)

character_count = len(project_input)
st.caption(f"{character_count:,} / {MAX_INPUT_CHARS:,} characters")

generate_button = st.button(
    "🤖 Generate AI Output",
    type="primary",
    use_container_width=True,
)

if generate_button:
    cleaned_input = project_input.strip()

    if not cleaned_input:
        st.error(
            "Please enter some project information before generating "
            "an output."
        )
        st.stop()

    if len(cleaned_input) > MAX_INPUT_CHARS:
        st.error(
            f"Your input is too long. Please shorten it to "
            f"{MAX_INPUT_CHARS:,} characters or fewer."
        )
        st.stop()

    with st.spinner("Generating the project-management output..."):
        try:
            output = generate_response(function, cleaned_input)

            st.subheader("3. AI-Generated Draft")

            st.markdown(output)

            st.warning(
                "⚠️ Verification required: Review the generated content "
                "against the original project information before using it. "
                "The model may produce plausible but unsupported information."
            )

            st.download_button(
                label="⬇️ Download output as text",
                data=output,
                file_name=(
                    function.lower()
                    .replace(" ", "_")
                    + "_ai_draft.txt"
                ),
                mime="text/plain",
            )

        except RuntimeError as exc:
            st.error(str(exc))

        except Exception:
            st.error(
                "An unexpected problem occurred while generating the output. "
                "Please try again."
            )

st.divider()

with st.expander("About this prototype"):
    st.write(
        "This prototype uses Qwen2.5-3B-Instruct through Hugging Face "
        "Inference Providers. It is designed as an AI-assisted project-"
        "management tool for exploratory academic evaluation."
    )
    st.write(
        "The five supported functions are Project Plan, Risk Register, "
        "Meeting Minutes, Project Summary and Stakeholder Communication."
    )
