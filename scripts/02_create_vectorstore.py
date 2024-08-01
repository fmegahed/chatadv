
EMBEDDINGS_DIRECTORY = './vstore' # directory to store embeddings

# Needed Libraries
# -----------------------------------------------------------------------------
import pickle

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reading the pickle MD file:
# -----------------------------------------------------------------------------
markdown_document = pickle.load(open("data/website_data_string_md.pkl", "rb"))

# Splitting the data into chunks based on the header # Document 1, # Document 2, etc.
# -----------------------------------------------------------------------------
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("# Document", 'Webpage')])
md_header_splits = markdown_splitter.split_text(markdown_document)

# Create a dict of webpage number and source from md_header_splits
webpage_source_dict = {}
for i, sublist in enumerate(md_header_splits):
    source = sublist.page_content.split('\n## Source: ')[1].split('\n')[0]
    webpage_source_dict[i] = source

# find the number of chrs in each sublist of md_header_splits
print([len(sublist.page_content) for sublist in md_header_splits])

# appluting text splitter per https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=7500, chunk_overlap=500
)
splits = text_splitter.split_documents(md_header_splits)

# Add \n## Source: to page contents if it doesn't already exist
for i, split in enumerate(splits):
    if '## Source:' not in split.page_content:
      j = splits[i].metadata['Webpage']
      j = int(j)
      splits[i].page_content = split.page_content + f'\n## Source: {webpage_source_dict[j]}\n' 



# Create the embeddings and the vectorstore
# -----------------------------------------------------------------------------
embeddings_model = OpenAIEmbeddings(model = 'text-embedding-3-small')

# get embeddings for the data and create the vectorstore
vectorstore = FAISS.from_documents(documents = splits, embedding=embeddings_model)

vectorstore.save_local(os.path.join(EMBEDDINGS_DIRECTORY, f'webpage_vectorstore'))
