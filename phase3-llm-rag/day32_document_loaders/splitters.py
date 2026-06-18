from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Gradient descent is an optimization algorithm used to minimize
a function by iteratively moving in the direction of steepest descent.
In machine learning, it is used to find the model parameters that
minimize the loss function. The learning rate controls how large each
step is during the descent process. If the learning rate is too large,
the algorithm may overshoot the minimum. If too small, convergence
will be very slow. Stochastic gradient descent (SGD) is a variant
that uses a single random sample or small batch to estimate the gradient.
Mini-batch gradient descent is a compromise between batch and stochastic
gradient descent."""

print("[CHUNK SIZE COMPARISON]\n")

for chunk_size in [100, 200, 300]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20
    )
    chunks = splitter.split_text(text)
    print(f"chunk_size={chunk_size:3d} -> {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk[:60]}...")
    print()

print("[KEY INSIGHT]")
print("chunk_overlap=20 means each chunk shares 20 chars with the next.")
print("This prevents answers from being split across chunk boundaries.")