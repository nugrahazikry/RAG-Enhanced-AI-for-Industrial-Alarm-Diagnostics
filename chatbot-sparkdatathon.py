<<<<<<< HEAD
import streamlit as st
import pandas as pd
from vertexai.generative_models import GenerativeModel, Tool
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
import os
import streamlit as st
from PIL import Image
import io
from vertexai import rag
import vertexai
from google.cloud import storage
from dotenv import load_dotenv
import json
from google.oauth2 import service_account
import subprocess

class RAGASVertexAIEmbeddings(VertexAIEmbeddings):
    """Wrapper for RAGAS"""

    async def embed_text(self, text: str) -> list[float]:
        """Embeds a text for semantics similarity"""
        return self.embed([text], 1, "SEMANTIC_SIMILARITY")[0]
    
def evaluate_faithfulness(dataset, metrics, max_iterations=5):
    iteration = 0
    faithfulness_score = 0
    
    while iteration < max_iterations:
        result_set = []
        for i in range(len(dataset)):
            result = evaluate(
                dataset=Dataset.from_dict(dataset[i : i + 1]),
                metrics=metrics,
                raise_exceptions=False,
            )
            result_set.append(result.to_pandas())
        
        results_df = pd.concat(result_set)
        faithfulness_score = results_df['faithfulness'].mean()
        
        if faithfulness_score >= 0.8:
            break
        
        iteration += 1
    
    return faithfulness_score

# Load GitHub Token from Environment Variable
GITHUB_TOKEN = st.secrets["environment"]["github_secret_key"]

REPO_URL = f"https://nugrahazikry:{GITHUB_TOKEN}@github.com/nugrahazikry/spark-itb-rag-industry-maintenance.git"

subprocess.run(["git", "clone", REPO_URL, "repo_folder"])

# Load environment variables from .env file
load_dotenv("environment.env")

# Set Page Configuration
st.set_page_config(layout="wide")

# gcp_keys = st.secrets["gcp_keys"]
gcp_keys = dict(st.secrets["gcp_keys"])  # Convert AttrDict to a regular dict

# Setting up credentials
credentials = service_account.Credentials.from_service_account_info(gcp_keys)

# Set up a temporary credential file
credentials_path = "/tmp/gcp_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Write the JSON credentials file
with open(credentials_path, "w") as f:
    json.dump(gcp_keys, f)

storage_client = storage.Client()

# Setting up important parameters
display_name = "pdf-maintenance-guidebook-corpus"
bucket_name = "sparkdatathon-keprof-reborn-cloud-storage"
folder_path = "maintenance-image-dataset"

# Preparing the Vertex AI
llm_model = VertexAI(model_name=st.secrets["environment"]["model_name"])
embeddings = VertexAIEmbeddings(model_name=st.secrets["environment"]["embedding_name"])
vertexai.init(project=st.secrets["environment"]["PROJECT_ID"], location=st.secrets["environment"]["LOCATION_ID"])

# Setting up RAG evaluation with RAGAS
embeddings = RAGASVertexAIEmbeddings(model_name="text-embedding-005")
ragas_llm = LangchainLLMWrapper(llm_model)
metrics = [faithfulness]

for m in metrics:
    # change LLM for metric
    m.__setattr__("llm", ragas_llm)

    # check if this metric needs embeddings
    if hasattr(m, "embeddings"):
        # if so change with Vertex AI Embeddings
        m.__setattr__("embeddings", embeddings)

# List down all the blobs inside cloud storage
bucket = storage_client.bucket(bucket_name)
blobs = bucket.list_blobs(prefix=folder_path)

# Initialize session state for storing question and answer
if "question" not in st.session_state:
    st.session_state.question = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "image_index" not in st.session_state:
    st.session_state.image_index = 0  # Default to the first image

# Create a corpus connection
def corpus_connector(question):

    # List the rag corpus you just created
    corpus_name = st.secrets["environment"]["corpus_name"]
    
    # Direct context retrieval
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=3,  # Optional
        filter=rag.Filter(vector_distance_threshold=0.5))

    # Get the response of the retrieval source document with grounding
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name,)],
        text=question, rag_retrieval_config=rag_retrieval_config,)
    
    rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(rag_corpus=corpus_name,)]
                ,rag_retrieval_config=rag_retrieval_config,),))
    
    # Create a gemini model instance
    rag_model = GenerativeModel(
        model_name=st.secrets["environment"]["model_name"], tools=[rag_retrieval_tool])

    # Prompting to extract the answer from question
    prompt = f"""
    "Based on the provided document(s), generate a detailed answer to the following question:
{question}

General instruction:
1. The answer must be as detailed as possible, strictly using the information from the given sources.
2. Ensure that the response contains the exact sentences from the document; do not paraphrase or alter the wording.
3. If the source includes bullet points for explanation, retain them as they are to maintain clarity and structure.
4. Do not add any external information or interpretation—stick strictly to the document content.
5. Make sure it is properly structure on the answer"""

    # Generate response
    response = rag_model.generate_content(prompt)
    response_text = response.text

    # Filtering source document
    titles = [
        chunk.retrieved_context.title
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks]
    
    contextual_texts = [
        chunk.retrieved_context.text
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks]
    
    jpg_titles = [title.replace('.pdf', '.jpg') for title in titles]

    return response_text, jpg_titles, contextual_texts

# Title
st.title("Let Vertex AI RAG assist your query")
st.write("Enter a question below to retrieve information from the maintenance documents.")

with st.container(border=True):
    # Ask a Question Section
    st.subheader("Ask a Question")
    st.session_state.question = st.text_input("Your Question", "")
    st.session_state.ask_button = st.button("Ask", use_container_width=True)

# Second Section: Answer and Image Display
if st.session_state.ask_button:
    # generating your answer
    st.session_state.answer, st.session_state.jpg_titles, st.session_state.contextual_texts = corpus_connector(st.session_state.question)
    
    # Generating image function
    # Initialize image list in session state
    if "images" not in st.session_state:
        st.session_state.images = []
    
    # Clear previous images
    st.session_state.images.clear()

    # Filter and verify only valid JPEG images
    maintenance_doc_list_image = []

    for blob in blobs:
        # Extract filename and check if it’s in our filter list
        filename = blob.name.split("/")[-1]
        if filename in st.session_state.jpg_titles:
            # Download and store the image
            image_data = blob.download_as_bytes()
            img = Image.open(io.BytesIO(image_data))
            st.session_state.images.append(img)

    # Ensure image index is within bounds
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0
    elif st.session_state.image_index >= len(st.session_state.images):
        st.session_state.image_index = 0

    # Generating RAGAS evaluation
    # Run the evaluation on every row of the dataset
    data_samples = {
        'question': [st.session_state.question],
        'answer': [st.session_state.answer],
        'contexts' : [st.session_state.contextual_texts],
    }
    dataset = Dataset.from_dict(data_samples)
    st.session_state.faithfulness_score = evaluate_faithfulness(dataset, metrics)

# Define the user to write the question before having answer
if st.session_state.answer == None:
    st.subheader("Please write your question first before moving on")

# Define the user if the answer is not available within the source document
elif st.session_state.answer != None and not st.session_state.jpg_titles:
    col1, col2 = st.columns([1, 3])  # Left section 25%, Right section 75%

    with col1:
        with st.container(border=True):
            st.subheader("Answer")
            st.write(st.session_state.answer)
            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("Faithfulness score")
            st.write(st.session_state.faithfulness_score)
    
    with col2:
        with st.container(border=True):
            st.subheader("I'm sorry, No documents are found for the selected answer.")

# Define the user if the answer is available within the source document
else:
    col1, col2 = st.columns([1, 3])  # Left section 25%, Right section 75%

    with col1:
        with st.container(border=True):
            st.subheader("Answer")
            st.write(st.session_state.answer)
            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("Faithfulness score")
            st.write(st.session_state.faithfulness_score)

    with col2:
        with st.container(border=True):
            st.subheader("Representation Image")

            if st.session_state.images and len(st.session_state.images) > 0:  # Ensure images exist
                if len(st.session_state.images) > 1:  # Only show slider if there are multiple images
                    new_image_index = st.slider(
                        "Scroll through images",
                        0,
                        len(st.session_state.images) - 1,
                        st.session_state.image_index,
                        key="image_slider"
                    )

                    # Update session state with new index
                    if new_image_index != st.session_state.image_index:
                        st.session_state.image_index = new_image_index
                else:
                    st.session_state.image_index = 0  # Ensure index is set for a single image

                # Display the selected image
                st.image(
                    st.session_state.images[st.session_state.image_index],
                    caption=f"Image {st.session_state.image_index + 1}",
                    use_column_width=True,
                )

            else:
                st.write("No documents are found for the selected answer.")
=======
import streamlit as st
import pandas as pd
from vertexai.generative_models import GenerativeModel, Tool
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
import os
import streamlit as st
from PIL import Image
import io
from vertexai import rag
import vertexai
from google.cloud import storage
from dotenv import load_dotenv
import json
from google.oauth2 import service_account
import subprocess

class RAGASVertexAIEmbeddings(VertexAIEmbeddings):
    """Wrapper for RAGAS"""

    async def embed_text(self, text: str) -> list[float]:
        """Embeds a text for semantics similarity"""
        return self.embed([text], 1, "SEMANTIC_SIMILARITY")[0]
    
def evaluate_faithfulness(dataset, metrics, max_iterations=5):
    iteration = 0
    faithfulness_score = 0
    
    while iteration < max_iterations:
        result_set = []
        for i in range(len(dataset)):
            result = evaluate(
                dataset=Dataset.from_dict(dataset[i : i + 1]),
                metrics=metrics,
                raise_exceptions=False,
            )
            result_set.append(result.to_pandas())
        
        results_df = pd.concat(result_set)
        faithfulness_score = results_df['faithfulness'].mean()
        
        if faithfulness_score >= 0.8:
            break
        
        iteration += 1
    
    return faithfulness_score

# Load GitHub Token from Environment Variable
GITHUB_TOKEN = st.secrets["environment"]["github_secret_key"]

REPO_URL = f"https://nugrahazikry:{GITHUB_TOKEN}@github.com/nugrahazikry/spark-itb-rag-industry-maintenance.git"

subprocess.run(["git", "clone", REPO_URL, "repo_folder"])

# Load environment variables from .env file
load_dotenv("environment.env")

# Set Page Configuration
st.set_page_config(layout="wide")

# gcp_keys = st.secrets["gcp_keys"]
gcp_keys = dict(st.secrets["gcp_keys"])  # Convert AttrDict to a regular dict

# Setting up credentials
credentials = service_account.Credentials.from_service_account_info(gcp_keys)

# Set up a temporary credential file
credentials_path = "/tmp/gcp_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Write the JSON credentials file
with open(credentials_path, "w") as f:
    json.dump(gcp_keys, f)

storage_client = storage.Client()

# Setting up important parameters
display_name = "pdf-maintenance-guidebook-corpus"
bucket_name = "sparkdatathon-keprof-reborn-cloud-storage"
folder_path = "maintenance-image-dataset"

# Preparing the Vertex AI
llm_model = VertexAI(model_name=st.secrets["environment"]["model_name"])
embeddings = VertexAIEmbeddings(model_name=st.secrets["environment"]["embedding_name"])
vertexai.init(project=st.secrets["environment"]["PROJECT_ID"], location=st.secrets["environment"]["LOCATION_ID"])

# Setting up RAG evaluation with RAGAS
embeddings = RAGASVertexAIEmbeddings(model_name="text-embedding-005")
ragas_llm = LangchainLLMWrapper(llm_model)
metrics = [faithfulness]

for m in metrics:
    # change LLM for metric
    m.__setattr__("llm", ragas_llm)

    # check if this metric needs embeddings
    if hasattr(m, "embeddings"):
        # if so change with Vertex AI Embeddings
        m.__setattr__("embeddings", embeddings)

# List down all the blobs inside cloud storage
bucket = storage_client.bucket(bucket_name)
blobs = bucket.list_blobs(prefix=folder_path)

# Initialize session state for storing question and answer
if "question" not in st.session_state:
    st.session_state.question = None
if "answer" not in st.session_state:
    st.session_state.answer = None
if "image_index" not in st.session_state:
    st.session_state.image_index = 0  # Default to the first image

# Create a corpus connection
def corpus_connector(question):

    # List the rag corpus you just created
    corpus_name = st.secrets["environment"]["corpus_name"]
    
    # Direct context retrieval
    rag_retrieval_config=rag.RagRetrievalConfig(
        top_k=3,  # Optional
        filter=rag.Filter(vector_distance_threshold=0.5))

    # Get the response of the retrieval source document with grounding
    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name,)],
        text=question, rag_retrieval_config=rag_retrieval_config,)
    
    rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(rag_corpus=corpus_name,)]
                ,rag_retrieval_config=rag_retrieval_config,),))
    
    # Create a gemini model instance
    rag_model = GenerativeModel(
        model_name=st.secrets["environment"]["model_name"], tools=[rag_retrieval_tool])

    # Prompting to extract the answer from question
    prompt = f"""
    "Based on the provided document(s), generate a detailed answer to the following question:
{question}

General instruction:
1. The answer must be as detailed as possible, strictly using the information from the given sources.
2. Ensure that the response contains the exact sentences from the document; do not paraphrase or alter the wording.
3. If the source includes bullet points for explanation, retain them as they are to maintain clarity and structure.
4. Do not add any external information or interpretation—stick strictly to the document content.
5. Make sure it is properly structure on the answer"""

    # Generate response
    response = rag_model.generate_content(prompt)
    response_text = response.text

    # Filtering source document
    titles = [
        chunk.retrieved_context.title
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks]
    
    contextual_texts = [
        chunk.retrieved_context.text
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks]
    
    jpg_titles = [title.replace('.pdf', '.jpg') for title in titles]

    return response_text, jpg_titles, contextual_texts

# Title
st.title("Let Vertex AI RAG assist your query")
st.write("Enter a question below to retrieve information from the maintenance documents.")

with st.container(border=True):
    # Ask a Question Section
    st.subheader("Ask a Question")
    st.session_state.question = st.text_input("Your Question", "")
    st.session_state.ask_button = st.button("Ask", use_container_width=True)

# Second Section: Answer and Image Display
if st.session_state.ask_button:
    # generating your answer
    st.session_state.answer, st.session_state.jpg_titles, st.session_state.contextual_texts = corpus_connector(st.session_state.question)
    
    # Generating image function
    # Initialize image list in session state
    if "images" not in st.session_state:
        st.session_state.images = []
    
    # Clear previous images
    st.session_state.images.clear()

    # Filter and verify only valid JPEG images
    maintenance_doc_list_image = []

    for blob in blobs:
        # Extract filename and check if it’s in our filter list
        filename = blob.name.split("/")[-1]
        if filename in st.session_state.jpg_titles:
            # Download and store the image
            image_data = blob.download_as_bytes()
            img = Image.open(io.BytesIO(image_data))
            st.session_state.images.append(img)

    # Ensure image index is within bounds
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0
    elif st.session_state.image_index >= len(st.session_state.images):
        st.session_state.image_index = 0

    # Generating RAGAS evaluation
    # Run the evaluation on every row of the dataset
    data_samples = {
        'question': [st.session_state.question],
        'answer': [st.session_state.answer],
        'contexts' : [st.session_state.contextual_texts],
    }
    dataset = Dataset.from_dict(data_samples)
    st.session_state.faithfulness_score = evaluate_faithfulness(dataset, metrics)

# Define the user to write the question before having answer
if st.session_state.answer == None:
    st.subheader("Please write your question first before moving on")

# Define the user if the answer is not available within the source document
elif st.session_state.answer != None and not st.session_state.jpg_titles:
    col1, col2 = st.columns([1, 3])  # Left section 25%, Right section 75%

    with col1:
        with st.container(border=True):
            st.subheader("Answer")
            st.write(st.session_state.answer)
            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("Faithfulness score")
            st.write(st.session_state.faithfulness_score)
    
    with col2:
        with st.container(border=True):
            st.subheader("I'm sorry, No documents are found for the selected answer.")

# Define the user if the answer is available within the source document
else:
    col1, col2 = st.columns([1, 3])  # Left section 25%, Right section 75%

    with col1:
        with st.container(border=True):
            st.subheader("Answer")
            st.write(st.session_state.answer)
            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("Faithfulness score")
            st.write(st.session_state.faithfulness_score)

    with col2:
        with st.container(border=True):
            st.subheader("Representation Image")

            if st.session_state.images and len(st.session_state.images) > 0:  # Ensure images exist
                if len(st.session_state.images) > 1:  # Only show slider if there are multiple images
                    new_image_index = st.slider(
                        "Scroll through images",
                        0,
                        len(st.session_state.images) - 1,
                        st.session_state.image_index,
                        key="image_slider"
                    )

                    # Update session state with new index
                    if new_image_index != st.session_state.image_index:
                        st.session_state.image_index = new_image_index
                else:
                    st.session_state.image_index = 0  # Ensure index is set for a single image

                # Display the selected image
                st.image(
                    st.session_state.images[st.session_state.image_index],
                    caption=f"Image {st.session_state.image_index + 1}",
                    use_column_width=True,
                )

            else:
                st.write("No documents are found for the selected answer.")
>>>>>>> 688669c (Initial commit)
