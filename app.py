import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
from openai import OpenAI

api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Define a function to extract text from PDF and generate a cover letter
def generate_cover_letter(company_name, position, job_description, pdf_file):
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load and process the PDF (applicant's resume)
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Combine all chunks into a single string
    resume_text = " ".join([doc.page_content for doc in texts])

    # Define the prompt template
    prompt = PromptTemplate(
    input_variables=["company_name", "position", "job_description", "resume"],
    template="""
    As an experienced professional resume writer, your task is to create a compelling and tailored cover letter for the {position} position at {company_name}. Use the provided job description and resume to craft a letter that highlights the candidate's most relevant qualifications and demonstrates their enthusiasm for the role.

    Job Description:
    {job_description}

    Candidate's Resume:
    {resume}

    Please write a cover letter that follows this structure:

    1. Opening Paragraph:
    - Address the hiring manager (use "Dear Hiring Manager" if no specific name is provided)
    - Mention the specific position and company name
    - Express enthusiasm for the role and briefly state how you learned about the opportunity

    2. Body Paragraph 1:
    - Highlight 2-3 key qualifications from the resume that directly align with the job requirements
    - Provide specific examples or achievements that demonstrate these qualifications

    3. Body Paragraph 2:
    - Discuss why you're interested in this specific role and company
    - Show that you've done research on the company by mentioning a recent accomplishment, project, or value that resonates with you

    4. Closing Paragraph:
    - Summarize why you would be a great fit for the position
    - Express eagerness for an interview
    - Thank the reader for their time and consideration

    5. Signature:
    - Use a professional closing (e.g., "Sincerely," or "Best regards,")
    - End with [Your Name]

    Additional Guidelines:
    - Keep the letter concise, ideally under 400 words
    - Use a professional yet engaging tone
    - Tailor the content specifically to the job description and company
    - Avoid repeating information verbatim from the resume
    - Proofread for any grammatical or spelling errors

    Cover Letter:
    """
)
    
    # Generate the cover letter
    formatted_prompt = prompt.format(
        company_name=company_name,
        position=position,
        job_description=job_description,
        resume=resume_text
    )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature = 0.5
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes cover letters."},
            {"role": "user", "content": formatted_prompt}
        ]
    )
    
    return response.choices[0].message.content

# Streamlit app
st.title("Cover Letter Generator")

company_name = st.text_input("Company Name")
position = st.text_input("Position")
job_description = st.text_area("Job Description", placeholder="Paste the job description here...")
pdf_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")

if st.button("Generate Cover Letter"):
    if company_name and position and job_description and pdf_file:
        with st.spinner("Generating cover letter..."):
            cover_letter = generate_cover_letter(company_name, position, job_description, pdf_file)
            st.text_area("Generated Cover Letter", value=cover_letter, height=300)
    else:
        st.warning("Please fill in all fields and upload a PDF resume.")
