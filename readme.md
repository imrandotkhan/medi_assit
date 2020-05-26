# MEDI ASSIST
==========================================================================================
## Steps to train and start using MEDI ASSIST for making inferences

* Make sure all the dependencies are installed.
* Run **train_bertffn.py** in your local or Colab. This will train our network to produce the embeddings that will be used for cosine similarity checks. Save model weights to the desired path.
* Re initialize the model once again to store sentence embeddings corresponding to the questions/answers locally. This can be done using **train_data_to_embedding.py**
* Once we have the embeddings, we can employee either elastic search, FAISS(Facebook AI Similarity Check), DiskANN(released late 2019 by Microsoft). This project uses **FAISS** which comes in both CPU/GPU flavour, very efficient code that uses concepts like BLAS, Multithreading SIMD...etc.
* During inference time, we do the tokenization of the query in accordance with bert and pass it through our network to generate the embedding. This embedding is then searched across the indexed dataset to find similar question and answer pairs as per the user need.
* Once we have fetched top k matching results, this can be sent to GPT2 estimator to properly curate the data in the most presentable format.

