from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("thesis.pdf")
pages = loader.load()

# Print first 3 pages raw text
for i in range(3):
    print(f"\n{'='*40}")
    print(f"PAGE {i+1}")
    print('='*40)
    print(pages[i].page_content)