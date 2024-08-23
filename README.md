# Cover Letter Generator

## Overview
This project is a Streamlit-based web application that generates tailored cover letters using OpenAI's GPT-3.5-turbo model. It takes into account the company name, position, job description, and the applicant's resume to create a personalized and compelling cover letter.

## Features
- User-friendly web interface built with Streamlit
- PDF resume parsing and text extraction
- Integration with OpenAI's GPT-3.5-turbo for natural language generation
- Customized cover letter generation based on job details and resume content

## Setup and Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-username/cover-letter-generator.git
   cd cover-letter-generator
   ```

2. Install the required packages:
   ```
   pip install streamlit langchain openai pypdf2
   ```

3. Set up your OpenAI API key:
   - Create a `.streamlit/secrets.toml` file in the project root
   - Add your OpenAI API key to this file:
     ```
     OPENAI_API_KEY = "your-api-key-here"
     ```

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open the provided URL in your web browser.

3. Fill in the required information:
   - Company Name
   - Position
   - Job Description
   - Upload your resume (PDF format)

4. Click the "Generate Cover Letter" button.

5. The generated cover letter will appear in the text area below.

## How It Works
1. The app takes user inputs for company name, position, and job description.
2. The user uploads their resume in PDF format.
3. The PDF is processed and text is extracted using LangChain's PyPDFLoader.
4. The extracted text is split into manageable chunks.
5. A prompt is generated using the job details and resume content.
6. The prompt is sent to OpenAI's GPT-3.5-turbo model to generate a tailored cover letter.
7. The generated cover letter is displayed to the user.

## Customization
You can modify the cover letter structure and guidelines by editing the prompt template in the `generate_cover_letter` function.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## Disclaimer
This tool is designed to assist in creating cover letters, but the output should be reviewed and adjusted as needed. Always personalize and verify the content before submitting a cover letter to potential employers.
