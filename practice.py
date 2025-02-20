# import os 

# from pypdf import PdfReader
# def load_document(file_path):
#     print("**********", os.path.basename(file_path),"**********")
#     print('===Loading Document===')
#     document_sentences = []
#     if file_path.endswith(".pdf"):
#         readed_pdf = PdfReader(file_path)
#         page_id = 0
#         for page in readed_pdf.pages:
#             document_sentences.append({os.path.basename(file_path)[:-4]+f'_page_{page_id}':page.extract_text().split('.')})
#             page_id += 1

#     elif file_path.endswith(".txt"):
#         with open(file_path) as file:
#             document_sentences.append({os.path.basename(file_path)[:-4]:file.read().split('.')})

#     return document_sentences


# print(load_document(r"D:\LLMs\RAG_Q&A session\pdfs\AttentionIsAllYouNeed.pdf"))


# def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=20):
#     chunks = []
#     start= 0 
#     while start<len(text):
#         end = start+chunk_size 
#         chunks.append(text[start:end])
#         start = end-chunk_overlap 
#     return chunks
# res_ch = split_text_into_chunks("""response : The life expectancy in the US is currently around 77 years.  However, it's important to note that this is an average and can vary significantly based on factors like race, gender, socioeconomic status, and geog
# important to note that this is an average and can vary significantly based on factors  (Centers for Disease Control and Prevention) and the SSA (Social Security Administration).
# like race, gender, socioeconomic status, and geographic location.  You'll find more precise and detailed data from sources like the CDC (Centers for Disease Control and Prevention) and the SSA (Social Security Administration).  """, chunk_size=100)



# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Whitespace

# # Sample data
# data = ["i", "am", "very", "happy", "today"]

# # Initialize BPE tokenizer
# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Whitespace()

# # Train BPE on sample data
# trainer = BpeTrainer(vocab_size=50, )#special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
# tokenizer.train_from_iterator(data, trainer)

# # Encode words using trained BPE tokenizer
# print('vocab : \n', [i for i in tokenizer.get_vocab().keys() if len(i)==1], "\n")
# print('vocab : \n', [i for i in tokenizer.get_vocab().keys() if len(i)==2], "\n")
# print('vocab : \n', [i for i in tokenizer.get_vocab().keys() if len(i)>2], "\n")
# print(f'{sum(tokenizer.get_vocab().values())}')
# print(f'len of vocab -- {tokenizer.get_vocab_size()}')
# for word in data:
#     print(f"{word} -> {tokenizer.encode(word).tokens}")

# print(len("AI Ethics is also a growing concern. Ensuring fairness, transparency, and accountability is crucial for responsible AI deployment."))

from langchain import hub 

prompt = hub.pull("wfh/proposal-indexing")
print(prompt)