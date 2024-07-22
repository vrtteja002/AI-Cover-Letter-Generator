import streamlit as st
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
import io
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def generate_cover_letter(company_name, position, job_description, resume_text):
    prompt = PromptTemplate(
        input_variables=["company_name", "position", "job_description", "resume"],
        template="""
        As an experienced professional resume writer, create a compelling cover letter for the {position} position at {company_name}. Use the provided job description and resume to highlight the candidate's relevant qualifications and enthusiasm.

        Job Description: {job_description}

        Candidate's Resume: {resume}

        Write a cover letter with:
        1. Opening: Address hiring manager, mention position and company, express enthusiasm.
        2. Body 1: Highlight 2-3 key qualifications aligning with job requirements. Provide examples.
        3. Body 2: Discuss interest in role and company. Show company research.
        4. Closing: Summarize fit, express interview eagerness, thank the reader.
        5. Signature: Professional closing and [Your Name].

        Guidelines:
        - Concise (under 400 words)
        - Professional yet engaging tone
        - Tailored to job and company
        - Avoid verbatim resume repetition
        - Proofread for errors

        Cover Letter:
        """
    )
    
    formatted_prompt = prompt.format(
        company_name=company_name,
        position=position,
        job_description=job_description,
        resume=resume_text
    )
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes cover letters."},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Streamlit app
st.title("Cover Letter Generator")

with st.form("cover_letter_form"):
    company_name = st.text_input("Company Name")
    position = st.text_input("Position")
    job_description = st.text_area("Job Description", placeholder="Paste the job description here...")
    pdf_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")
    
    submitted = st.form_submit_button("Generate Cover Letter")

if submitted and company_name and position and job_description and pdf_file:
    with st.spinner("Generating cover letter..."):
        resume_text = extract_text_from_pdf(pdf_file)
        cover_letter = generate_cover_letter(company_name, position, job_description, resume_text)
        st.text_area("Generated Cover Letter", value=cover_letter, height=400)
elif submitted:
    st.warning("Please fill in all fields and upload a PDF resume.")
