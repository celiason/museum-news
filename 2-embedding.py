import PyPDF2
import requests
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pymupdf4llm

# from data_processing import DataProcessing
# """Extracts data from pdfs and create jsonl"""

# if __name__ == "__main__":
#     data_processor = DataProcessing()
#     data_processor.execute_data_processing()

# data_dict = process_dir(path=, maxchunk=, minchunk=)
# embedding_dict_to_dataframe(data_dict)

from langchain.text_splitter import CharacterTextSplitter

text = """
Character splitting is the most basic form of splitting up your text.
It is the process of simply dividing your text into N-character sized chunks regardless of their content or form.
"""

# Function to chunk text
def chunk_text(text, chunk_size=512, overlap=25):
    """Splits the input text into chunks based on the specified chunk size.

    Args:
    - text (str): The input text to be chunked.
    - chunk_size (int): The maximum size of each chunk in terms of characters.

    Returns:
    - chunks (list): A list of chunks where each chunk is a string not
    exceeding the chunk size.
    """
    
    if overlap == 0:
        chunks = []
        words = text.split()
        current_chunk = ''
        for word in words:
            if len(current_chunk) + len(word) + 1 <= chunk_size:
                current_chunk += word + ' '
            else:
                chunks.append(current_chunk)
                current_chunk = word + ' '
        if current_chunk:
            chunks.append(current_chunk)
    
    else:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separator=' ', strip_whitespace=False)
        chunks = text_splitter.create_documents([text])
        chunks = [x.page_content for x in chunks]

    return chunks

# chunk_text(text, chunk_size=10, overlap=5)


# Function that reads in PDFs and creats chunks of text
def chunk_pdfs(file_path, chunk_size=2000, overlap=25, reader='pymupdf4llm'):
    with open(file_path, 'rb') as file:
        if reader == 'pymupdf4llm':
            text = pymupdf4llm.to_markdown(file, show_progress=False)
            chunks_info = chunk_text(text, chunk_size)
        if reader == 'pypdf2':
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                chunks_info = []
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                # Split text into chunks approximately the size of a page
                chunks = chunk_text(text, chunk_size, overlap=overlap)
                chunks_info.extend(chunks)
    return chunks_info

# Testing zone
# chunks = chunk_pdfs(file_path='pdfs/25720.pdf', chunk_size=512, reader='pymupdf4llm')
# chunks

# Loop through all PDF files and create chunks of text
dir_path = './pdfs'
unique_id = 1
chunk_size = 512 # optimum according to https://arxiv.org/pdf/2407.01219
chunk_overlap = 25 # ~from https://arxiv.org/pdf/2407.01219 (they used 20)
min_chunk_size = 50 # not sure needed?
files_list = []
for filename in tqdm(os.listdir(dir_path), desc="Processing PDF files"):
    if filename.endswith('.pdf'):
        file_path = os.path.join(dir_path, filename)
        chunks_info = chunk_pdfs(file_path, chunk_size, chunk_overlap)
        for chunk in chunks_info:
            file_info = {
                'id': unique_id,
                'title': filename,
                'text_chunk': chunk
            }
            if len(chunk) > min_chunk_size:
                files_list.append(file_info)
                unique_id += 1

# Embed chunks in vector space
ST = SentenceTransformer('all-MiniLM-L6-v2')

# Setup dataframe with doc IDs, text, etc.
train = pd.DataFrame.from_records(files_list, index='id')
train['new_column'] = train.title + ' - ' + train.text_chunk.str.lower()

# Get list of raw text chunks
texts = list(train.new_column)

# Loop through all chunks (by given step size) and create embeddings
output_list = []
step = 50
from tqdm import tqdm
for i in tqdm(range(0, len(texts), step), desc="Encoding slices"):
    current_slice = texts[i:i + step]
    output = ST.encode(current_slice)
    output_list.extend(output)

# Convert to dataframe
embeddings = pd.DataFrame(output_list)

# Merge with text chunk/doc metadata
train_fin = pd.concat([train, embeddings], axis=1)
train_fin.reset_index(inplace=True)

# Output for modeling
train_fin.to_csv("train.csv", index=False)

