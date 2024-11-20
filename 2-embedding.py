import PyPDF2
import requests
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pymupdf4llm
from langchain.text_splitter import CharacterTextSplitter

# from data_processing import DataProcessing
# """Extracts data from pdfs and create jsonl"""

# if __name__ == "__main__":
#     data_processor = DataProcessing()
#     data_processor.execute_data_processing()

# data_dict = process_dir(path=, maxchunk=, minchunk=)
# embedding_dict_to_dataframe(data_dict)

text = """
Character splitting is the most basic form of splitting up your text.
It is the process of simply dividing your text into N-character sized chunks regardless of their content or form.
"""

# Function to chunk text
def chunk_text(text, chunk_size=512, chunk_overlap=20, clean=True):
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
        # text = "hi [there]"
        chunks = text_splitter.create_documents([text])
        chunks = [x.page_content for x in chunks]
        if clean:
            chunks = [x.replace('[', ' ') for x in chunks]
            chunks = [x.replace(']', ' ') for x in chunks]

    return chunks

# chunk_text("hi [there]", chunk_size=10, overlap=5)

# Function that reads in PDFs and creats chunks of text
def chunk_pdfs(file_path, chunk_size=2000, chunk_overlap=20, reader='pymupdf4llm'):
    with open(file_path, 'rb') as file:
        if reader == 'pymupdf4llm':
            text = pymupdf4llm.to_markdown(file, show_progress=True, page_chunks=True)
            chunks_info = chunk_text(text, chunk_size)
        if reader == 'pypdf2':
            pdf_reader = PyPDF2.PdfReader(file)
            chunks_info = []
            for page_num in range(len(pdf_reader.pages)):    
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                # Split text into chunks approximately the size of a page
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                chunks_info.extend(chunks)
        if reader == 'pymupdf':
            # file='pdfs/25720.pdf'
            pdf_reader = pymupdf.open(file)
            chunks_info = []
            for page_num in range(len(pdf_reader)):    
                text = pdf_reader[page_num].get_text()
                # Split text into chunks approximately the size of a page
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                chunks_info.extend(chunks)
    return chunks_info

# res = pymupdf4llm.to_markdown('pdfs/25720.pdf', pages=[2], show_progress=True, page_chunks=True, margins=(0,0,0,0))
# print(res[0]['text'])

# Make images showing different extraction methods-

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def text_to_image(text, font_path="Arial", wrap=80, font_size=20, color=(0, 0, 0), bg=(255, 255, 255)):
    """Converts a chunk of text to an image."""

    text = text.replace('\n', '')
    
    wrapper = textwrap.TextWrapper(width=wrap)
    
    # filled_text = wrapper.fill(text=text)
    text_lines = wrapper.wrap(text=text)

    font = ImageFont.truetype(font_path, font_size)
    
    line_sizes = np.array([font.getbbox(x) for x in text_lines])
    height = sum(line_sizes[:, 3])
    width = max(line_sizes[:, 2])

    img = Image.new("RGB", (width, height), color='white')

    draw = ImageDraw.Draw(img)

    draw.multiline_text((0, 0), '\n'.join(text_lines), font=font, fill='black')

    return img

text_method1 = pymupdf4llm.to_markdown('pdfs/25720.pdf', pages=[0])
print(textwrap.fill(text_method1))
img1 = text_to_image(text_method1[0:800], font_size=14, wrap=40)
img1.save("text_pymupdf4llm.png")

text_method2 = pymupdf.open('pdfs/25720.pdf')[0].get_text()
img2 = text_to_image(text_method2[0:800], font_size=14, wrap=40)
img2.save("text_pymupdf.png")
                
text_method3 = PyPDF2.PdfReader('pdfs/25720.pdf').pages[0].extract_text()
img3 = text_to_image(text_method3[0:800], font_size=14, wrap=40)
img3.save("text_pypdf2.png")



# import pymupdf
# doc = pymupdf.open("pdfs/25720.pdf") # open a document
# print(textwrap.fill(doc[0].get_text()))
# len(doc)
# for page in doc: # iterate the document pages
#   text = page.get_text() # get plain text encoded as UTF-8


# Testing zone
# chunks = chunk_pdfs(file_path='pdfs/25720.pdf', chunk_size=512, reader='pymupdf')
# print(textwrap.fill(chunks[0]))
# chunks = chunk_pdfs(file_path='pdfs/25720.pdf', chunk_size=512, reader='pypdf2')
# chunks
# import textwrap
# print(textwrap.fill(chunks[7]))

# Loop through all PDF files and create chunks of text
dir_path = './pdfs'
unique_id = 1
chunk_size = 512 # optimum according to https://arxiv.org/pdf/2407.01219
chunk_overlap = 20 # ~from https://arxiv.org/pdf/2407.01219 (they used 20)
min_chunk_size = 50 # not sure needed?
files_list = []
for filename in tqdm(os.listdir(dir_path), desc="Processing PDF files"):
    if filename.endswith('.pdf'):
        file_path = os.path.join(dir_path, filename)
        chunks_info = chunk_pdfs(file_path, chunk_size, chunk_overlap, reader='pymupdf')
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
# ST = SentenceTransformer('BAAI/bge-small-en-v1.5')
# ST = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

# Setup dataframe with doc IDs, text, etc.
train = pd.DataFrame.from_records(files_list, index='id')
train['new_column'] = train.title + ' - ' + train.text_chunk.str.lower()

# Get list of raw text chunks
texts = list(train.new_column)

print(textwrap.fill(texts[110]))

# Loop through all chunks (by given step size) and create embeddings
output_list = []
step = 50
from tqdm import tqdm
for i in tqdm(range(0, len(texts), step), desc="Embedding chunks"):
    current_slice = texts[i : i + step]
    output = ST.encode(current_slice)
    output_list.extend(output)

# Convert to dataframe
embeddings = pd.DataFrame(output_list)

# Merge with text chunk/doc metadata
train_fin = pd.concat([train, embeddings], axis=1)
train_fin.reset_index(inplace=True)

# print(textwrap.fill(train_fin.loc[1, 'new_column']))

train_fin.shape

# Output for modeling
train_fin.to_csv("train.csv", index=False)

# import pandas as pd
# train = pd.read_csv("train.csv")
# train.columns
# train.loc[1, "title"]
# print(train.loc[1, "text_chunk"])

