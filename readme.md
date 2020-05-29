# MEDI ASSIST
==========================================================================================
## Steps to train and use MEDI ASSIST for making inferences

* Make sure all the dependencies are installed.
* Run **train_bertffn.py** in your local or Colab. This will train our network to produce the embeddings that will be used for cosine similarity checks. Save model weights to the desired path.
* Re initialize the model once again to store sentence embeddings corresponding to the questions/answers locally. This can be done using **train_data_to_embedding.py**
* Once we have the embeddings, we can employee either elastic search, FAISS(Facebook AI Similarity Check), DiskANN(released late 2019 by Microsoft). This project uses **FAISS** which comes in both CPU/GPU flavour, very efficient code that uses concepts like BLAS, Multithreading SIMD...etc.
* During inference time, we do the tokenization of the query in accordance with bert and pass it through our network to generate the embedding. This embedding is then searched across the indexed dataset to find similar question and answer pairs as per the user need.
* Once we have fetched top k matching results, this can be sent to GPT2 estimator to properly curate the data in the most presentable format.

## Data Pipeline
* Store Questions and Answers in csv file with all the question and its corresponding answer listed against each other.
* If a question has multiple answers then repeat the same question multiple times with different answers.
* We use BioBert as a pre-trained model for this application, hence get the tokenizer instance from biobert and pass it to the function that creates train/test features.
* Function **create_dataset_for_bert** under **dataset.py** performs following pre-processing tasks along with other optimizations for optimal and efficient training:
    * Train/Test split
    * Tokenization
    * Converting data into TFRecord and tf.Example, this makes reading data efficient and helps to serialize the data and store it in set of files(100-200 MB each) that can each be read linearly. This helps a lot when the data is being streamed over network. It also helps in caching any data-preprocessing.
    * Using **tf.data.experimental.bucket_by_sequence_length** that helps elements of the Dataset to be grouped together by length and then are padded and batched.
* We append "[CLS]" and "[SEP]" at the begining and end of each sentence. Then the entire sentence is tokenized and encoded.
* We create segment_id(contains all 0's, this is of the same length as that of sentence + 2 to account for CLS and SEP tokens. This signifies no. of sentences in our input text), input_id(stores id's from biobert vocab for the corresponding token in sentence), mask_id(has all 1's for real token and 0's for padded token).
* Finally a generator yeilds q_features+a_features and '1'(acts as label) which then gets converted to dense features using tf.sparse_to_dense followed by bucketing and rest of the steps mentioned above.
* During training we get the output generated from Biobert model from its last layer which is of the dimension [batchsize, sentence length, 768 size embeddings for each token in sentence] and take the mean across axis 1 to squish this output to size [batchsize, 768].
* Previous output is then fed into the feed forward netword followed by Softmax layer and then Categorical Cross Entropy loss.
* We use Accuracy as the performance metric.