import os 

from pypdf import PdfReader
def load_document(file_path):
    print("**********", os.path.basename(file_path),"**********")
    print('===Loading Document===')
    document_sentences = []
    if file_path.endswith(".pdf"):
        readed_pdf = PdfReader(file_path)
        page_id = 0
        for page in readed_pdf.pages:
            document_sentences.append({os.path.basename(file_path)[:-4]+f'_page_{page_id}':page.extract_text().split('.')})
            page_id += 1

    elif file_path.endswith(".txt"):
        with open(file_path) as file:
            document_sentences.append({os.path.basename(file_path)[:-4]:file.read().split('.')})

    return document_sentences


print(load_document(r"D:\LLMs\RAG_Q&A session\pdfs\AttentionIsAllYouNeed.pdf"))