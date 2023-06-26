from config import PINECONE_API_KEY, \
PINECONE_REGION, PINECONE_INDEX_NAME, MODEL_NAME, MODEL_NAME_CLIP, \
aws_access_key_id, aws_secret_access_key

from PyPDF2 import PdfReader
import pinecone
from sentence_transformers import SentenceTransformer
import os
import boto3
from io import BytesIO

def get_pdf_data(file_path, num_pages = 1):
    reader = PdfReader(file_path)
    full_doc_text = ""
    pages = reader.pages
    num_pages = len(pages) 
    
    try:
        for page in range(num_pages):
            current_page = reader.pages[page]
            text = current_page.extract_text()
            full_doc_text += text
    except:
        print("Error reading file")
    finally:
        return full_doc_text
    
def get_chunks(fulltext:str,chunk_length =500) -> list:
    text = fulltext

    chunks = []
    while len(text) > chunk_length:
        last_period_index = text[:chunk_length].rfind('.')
        if last_period_index == -1:
            last_period_index = chunk_length
        chunks.append(text[:last_period_index])
        text = text[last_period_index+1:]
    chunks.append(text)

    return chunks

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_REGION)
index = pinecone.Index(PINECONE_INDEX_NAME)

model = SentenceTransformer(MODEL_NAME)
length = 512 - 384 
list_custom = [1] * length


def addData(corpusData,filename,CATEGORY):
    id  = index.describe_index_stats()['total_vector_count']
    for i in range(len(corpusData)):
        chunk=corpusData[i]
        embedding = model.encode(chunk).tolist()
        embedding_all = embedding + list_custom
        chunkInfo=(str(id+i),
                embedding_all,
                {'sentence': chunk,'category':CATEGORY,'filename':filename})
        index.upsert(vectors=[chunkInfo])

def insert_pinecone(filename,CATEGORY):
    print("Processing file: ", filename)
    full_doc_text = get_pdf_data(filename)
    Lines = get_chunks(full_doc_text)
    print("Number of chunks: ", len(Lines))
    addData(Lines,filename,CATEGORY)

def insert_pinecone_file_path(file_path,CATEGORY="ncert"):
    files = os.listdir(file_path)
    for filename in files:
        if filename.endswith(".pdf"):
            insert_pinecone(file_path + "/" + filename,CATEGORY)

def insert_pinecone_S3(s3_bucket_name,CATEGORY="ncert"):
    # Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
                    aws_secret_access_key=aws_secret_access_key)
    for key in s3.list_objects(Bucket=s3_bucket_name)['Contents']:
        filename = key['Key']
        if filename.endswith(".pdf"):
            
            response = s3.get_object(Bucket=s3_bucket_name, Key=filename)

            # Read the content of the PDF file
            pdf_content = response['Body'].read()
            pdf_reader = PdfReader(BytesIO(pdf_content))
            # Read the text from each page of the PDF
            text_content = ''
            for page in pdf_reader.pages:
                text_content += page.extract_text()
            Lines = get_chunks(text_content)
            print("Number of chunks: ", len(Lines))
            addData(Lines,filename,CATEGORY)