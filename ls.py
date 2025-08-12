# !pip install streamlit
# !pip install pdfplumber
# !pip install chromadb

# set up gemini
from google import generativeai as genai

GOOGLE_API_KEY="GEMINI_API_KEY"

genai.configure(api_key=GOOGLE_API_KEY)
model=genai.GenerativeModel("gemini-2.0-flash")


#chromadb-it is an open-source AI-native vector database designed for efficient
#storage and retrieval of embeddings, documents, and metadata. It supports features
#like vector search, full-text search, and multimodal retrieval, making it ideal for
#building applications powered by large language models.

#set up chromadb
import chromadb
from chromadb.utils import embedding_functions
from chromadb import Documents , EmbeddingFunction , Embeddings

DB_NAME="lectures_notes"

#this is the customize class for:-
#--> task type, model version, etc.
#--> see exactly how the request is made
#--> Uses gemini-pro and embed_content() manually

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self,input: Documents)-> Embeddings:
    response= genai.embed_content(
        model="models/embedding-001",
        content=input,
        task_type="retrieval_document",
        title="Lecture Notes"
    )
    return response["embedding"]

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode=True

chroma_client = chromadb.Client()
db=chroma_client.get_or_create_collection(name=DB_NAME,embedding_function=embed_fn)

#pdfplumber -is a powerful Python library used for extracting text, tables, and
#other data from PDF documents with high accuracy. It allows users to navigate
#through pages, detect structured data, and efficiently parse information without
#manual copying.

# Function to extract text from pdf
import pdfplumber

def extract_text_from_pdf(uploaded_file):
  text=""
  with pdfplumber.open(uploaded_file) as pdf:
    for page in pdf.pages:
      text+=page.extract_text() + "\n"
  return text

#Chunk tect for embeddings

def chunk_text(text,chunk_size=500):
  words=text.split()
  return [" ".join(words[i:i+chunk_size]) for i in range(0,len(words),chunk_size)]

#Add chunks to chromadb

def add_to_vector_db(chunks):
  for i ,chunk in enumerate(chunks):
    db.add(documents=[chunk],ids=[f"chunk_{i}"])

# 1) first function that my project can do is:-
# summarising the lecture notes using gemini

#Generate summary using Gemini
def generate_summary(text):
  prompt = f"""
  Summarize the following notes in the clear manner:

  {text}
  """
  response=model.generate_content(prompt)
  return response.text.strip()

# 2) second function that my project can do is:-

# generating the quiz using gemini in which:-
# --> model create 5 quizes each contain 4 options and out of which 1 is correct
#--> if user can select the option
#--> if the selected option is incorrect the model provide the correct answer with some explanation
#--> you get +1 score for the correct option selected

# generating the 3 flashcards with term - definition format

# Generate quiz and flashcard with Gemini

def generate_quiz_flashcards(summary):
  prompt =f"""
  Based on this lecture summary ,generate :
  - 5 multiple choice questions (each with 4 options and 1 correct answer)
  - 3 flashcards with term - definition format
  - Include a short explanation for each correct answer
  Format your output as JSON:
  {{
      "mcqs": [
           {{
                "questions": "....",
                "options": ["....","....","....","...."],
                "answer": "...."
                "explanation": "...."
           }},
           ...
      ],
      "flashcards":[
        {{"term": "...." , "definition": "...." ,}},
        ...
      ]
  }}

  Summary:
  {summary}
  """
  response=model.generate_content(prompt)
  try:
    json_block = json.loads(response.text.strip().split("```json")[-1].split("```")[-2])
    return json_block
  except:
    return json.loads(response.text)

# 3) third function that my project can do is:-

# A chatbot which help you to ask question from the lecture notes to clear the
#doubts related to any topic present in the pdf

# chatbot logic

def chatbot(query):
  results=db.query(query_texts=[query],n_results=3)
  context="\n".join([doc for doc in results["documents"][0]])
  prompt =f"""
  Based on the following context ,answer the question:

  {context}

  Question:{query}
  """
  response=model.generate_content(prompt)
  return response.text.strip()

#Streamlit UI

import streamlit as st
import json
import os
import tempfile

def main():
    st.title("📚 Lecture Summarizer + Quiz Generator + Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text....."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)
            add_to_vector_db(chunks)

        # Initialize session state for summary and quiz if not present
        if "summary" not in st.session_state:
            st.session_state.summary = None
            st.session_state.quiz_flashcards = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False

        # Generate button that saves results to session state
        if st.button("Generate Summary and Quiz and Flashcards"):
            with st.spinner("Generating summary and quiz....."):
                st.session_state.summary = generate_summary(text)
                st.session_state.quiz_flashcards = generate_quiz_flashcards(st.session_state.summary)
                st.session_state.quiz_answers = {}  # Reset answers
                st.session_state.quiz_submitted = False

        # Display summary if available
        if st.session_state.summary:
            st.subheader("📌 Summary")
            st.write(st.session_state.summary)

        # Display quiz if available
        if st.session_state.quiz_flashcards:
            st.subheader("📝 Quiz")
            
            # Create the quiz form
            with st.form(key="quiz_form"):
                for idx, q in enumerate(st.session_state.quiz_flashcards["mcqs"], 1):
                    st.markdown(f"**Q{idx}. {q['question']}**")
                    
                    # Radio buttons for each question
                    answer = st.radio(
                        f"Choose your answer for Q{idx}:",
                        q["options"],
                        key=f"q{idx}",
                        index=None
                    )
                    
                    # Store answer in session state
                    if answer:
                        st.session_state.quiz_answers[idx] = answer
                    
                    st.markdown("---")  # Separator
                
                # Single submit button for the whole form
                submit_quiz = st.form_submit_button("Submit Quiz")
                
                if submit_quiz:
                    st.session_state.quiz_submitted = True
            
            # Display results after submission (outside the form)
            if st.session_state.quiz_submitted:
                st.subheader("Quiz Results")
                correct_count = 0
                
                for idx, q in enumerate(st.session_state.quiz_flashcards["mcqs"], 1):
                    user_answer = st.session_state.quiz_answers.get(idx)
                    
                    if user_answer:
                        if user_answer == q["answer"]:
                            st.success(f"Q{idx}: Correct! 🎉")
                            correct_count += 1
                        else:
                            st.error(f"Q{idx}: Incorrect. The correct answer is: {q['answer']}")
                        st.info(f"Explanation: {q['explanation']}")
                    else:
                        st.warning(f"Q{idx}: Not answered")
                        st.info(f"The correct answer is: {q['answer']}")
                        st.info(f"Explanation: {q['explanation']}")
                st.markdown(f"### Your Score: {correct_count}/{len(st.session_state.quiz_flashcards['mcqs'])}")
                
                if st.button("Try Again"):
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.experimental_rerun()

            # Display flashcards
            if st.session_state.quiz_flashcards:
                st.subheader("🔖 Flashcards")
                for flashcard in st.session_state.quiz_flashcards["flashcards"]:
                    st.markdown(f"**Term:** {flashcard['term']}")
                    st.markdown(f"**Definition:** {flashcard['definition']}")
                    st.markdown("---")

    # Chatbot section 
    st.markdown("---")
    st.subheader("💬 Ask the question about the lecture")


    if "messages" not in st.session_state:
        st.session_state.messages=[]

    user_query=st.text_input("Enter the question")

    if st.button("Ask") and user_query:
        if user_query.lower() in ["exit","quit","stop"]:
            st.success("👋 Conversation ended. Feel free to ask another question!")
        else:
            with st.spinner("Thinking..."):
                answer=chatbot(user_query)

            st.session_state.messages.append({"role":"user","content":user_query})
            st.session_state.messages.append({"role":"bot","content":answer})

    for msg in st.session_state.messages:
        if msg["role"] =="user":
            st.markdown(f"🧑‍🎓 **You**: {msg['content']}")
        else:
            st.markdown(f"🤖 **Bot**: {msg['content']}")

    if st.button("Reset Conversation"):
        st.session_state.messages=[]
        st.success("🧹 Conversation cleared.")

if __name__ == "__main__":
    main()
