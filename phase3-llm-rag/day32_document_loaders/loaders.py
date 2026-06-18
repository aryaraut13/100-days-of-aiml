import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Create a sample text file to load
sample_text = """
Machine Learning Fundamentals

Chapter 1: Supervised Learning
Supervised learning is a type of machine learning where the algorithm learns
from labeled training data. The algorithm learns a mapping from inputs to
outputs based on example input-output pairs. Common supervised learning
algorithms include linear regression, logistic regression, decision trees,
random forests, and support vector machines.

Chapter 2: Unsupervised Learning
Unsupervised learning involves finding patterns in data without labeled
responses. The algorithm must discover the underlying structure in the data
on its own. Common unsupervised learning techniques include clustering
algorithms like K-means and DBSCAN, dimensionality reduction methods like
PCA and t-SNE, and generative models like GANs and VAEs.

Chapter 3: Model Evaluation
Model evaluation is crucial for understanding how well a machine learning
model performs on unseen data. Key metrics include accuracy, precision,
recall, F1 score, and AUC-ROC. Cross-validation techniques help ensure
that evaluation is robust and not dependent on a single train-test split.
Overfitting and underfitting are the two main failure modes to watch for.

Chapter 4: Feature Engineering
Feature engineering is the process of using domain knowledge to create
features that make machine learning algorithms work better. This includes
handling missing values, encoding categorical variables, scaling numeric
features, and creating new features from existing ones. Good feature
engineering often has more impact than algorithm selection.
"""

# Save sample text
with open("sample_doc.txt", "w") as f:
    f.write(sample_text)

# Load the document
loader = TextLoader("sample_doc.txt")
docs   = loader.load()

print(f"[LOADED] {len(docs)} document(s)")
print(f"         Total characters: {len(docs[0].page_content)}")
print(f"         Preview: {docs[0].page_content[:100]}...\n")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
)

chunks = splitter.split_documents(docs)

print(f"[SPLIT]  {len(chunks)} chunks created")
print(f"         chunk_size=300, chunk_overlap=50\n")

for i, chunk in enumerate(chunks[:4]):
    print(f"Chunk {i+1} ({len(chunk.page_content)} chars):")
    print(f"  {chunk.page_content[:120]}...")
    print()