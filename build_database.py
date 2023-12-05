import milvus
from transformers import AutoTokenizer, AutoModel
import PyPDF2

# Connect to the Milvus server
milvus_client = milvus.Milvus(host='localhost', port='19530')

# Connect to the Milvus collection
collection_name = 'pdf_collection'
dimension = 768
milvus_client.create_collection({'collection_name': collection_name, 'dimension': dimension})

# Load PDF documents and extract text
pdf_files = ['file1.pdf', 'file2.pdf', 'file3.pdf']
texts = []
for file in pdf_files:
    with open(file, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        texts.append(text)

# Initialize BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Insert data into the Milvus collection
vectors = []
for text in texts:
    # Preprocess and tokenize the text
    processed_text = preprocess_text(text)
    tokens = tokenizer.encode_plus(processed_text, add_special_tokens=True, padding='max_length', truncation=True,
                                   max_length=128, return_tensors='pt')

    # Generate BERT embeddings
    embeddings = model(**tokens)[0][:, 0, :].numpy()

    # Insert vectors into the Milvus collection
    vectors.append(embeddings)

vectors = np.array(vectors)
milvus_client.insert(collection_name=collection_name, records=vectors)

# Disconnect from the Milvus server
milvus_client.close_connection()
