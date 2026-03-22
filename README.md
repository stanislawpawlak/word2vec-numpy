# Word2Vec in NumPy

This repository contains a from-scratch implementation of the **Word2Vec** algorithm (Skip-Gram with Negative Sampling variant) using pure Python and NumPy. 

The core optimization loop, including the forward pass, loss calculation, backpropagation, and parameter updates, is implemented entirely without the use of high-level ML frameworks like PyTorch or TensorFlow. This project serves as a deep dive into the underlying mathematics of word embeddings.

##  References & Mathematical Foundation
The architectural choices, loss function, and gradient derivations for this implementation are heavily based on the materials from Stanford University's CS224n course:
* **Algorithm Math & Loss Function:** [CS224n Notes 01 - Word Vectors I: Introduction, SVD and Word2Vec](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)
* **Cosine Similarity Evaluation:** [Cosine Similarity: How does it measure the similarity? Maths behind and usage in Python](https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db/)

##  Core Architecture

1. **Custom Data Preprocessor:**
   * Cleans text and filters out custom stop words.
   * Builds a vocabulary dictionary (`word2id`, `id2word`) mapping based on a minimum frequency threshold.
   * Generates context-target training pairs using a sliding window.
   * Implements the **3/4 power trick** for the negative sampling distribution to slightly increase the probability of sampling less frequent words. Additionally, we filter out exceptionally rare occurrences (noise), allowing the model to better capture deep semantic dependencies without overfitting to anomalies.

2. **Word2Vec Model (Skip-Gram + Negative Sampling):**
   * **Forward Pass:** Computes the dot product between the central word vector and context/negative word vectors, mapped through a numerical-stable Sigmoid function.
   * **Loss Function:** Calculates the Negative Sampling Loss (minimizing the distance to true context words while maximizing the distance to noise words).
   * **Backpropagation:** Computes gradients for both the output matrix ($U$) and the input matrix ($V$) using the chain rule.
   * **Optimization:** Updates parameters via Stochastic Gradient Descent (SGD).

3. **Semantic Evaluation (Cosine Similarity):**
   * Implements the standard Cosine Similarity formula: `(A · B) / (||A|| * ||B||)`.
   * Includes methods to find the `top_n` most similar and least similar words to evaluate the learned semantic space.

##  Libraries
* `numpy` (for matrix operations, backpropagation, and core mathematics).
* `gensim.downloader` (*Note: Used strictly as an API to download the `text8` dataset quickly. No ML framework was used for the training process itself.*)

##  Dataset (`text8`)
The model is trained on the classic **`text8`** dataset, widely used in the NLP community for quick prototyping of word embedding models. 

* **Source:** A heavily cleaned, 100MB extract from English Wikipedia (created by Matt Mahoney).
* **Format:** Strictly lowercase letters (`a-z`) and spaces. All punctuation, numbers, and special characters have been removed.
* **Usage in this project:** To ensure rapid training without GPU acceleration, the script processes a subset of the data (the first 21 chunks of text, roughly ~210,000 words), which generates approximately 875,000 sliding-window training pairs. This subset is sufficient to prove the model's ability to learn distinct semantic relationships.

##  Training & Sample Output
After training on the aforementioned subset for 10 epochs, the loss steadily decreased from ~3.18M to ~1.98M. 

The model successfully captures logical, semantic, and historical relationships. Here is the actual evaluation output:

```text
VECTOR SIMILARITY EVALUATION
========================================

WORD: **KING**
   TOP SIMILAR: 
      0.5572 | coronis          
      0.5067 | thessaly         
      0.5063 | cyparissus       
      0.4907 | priam            
      0.4876 | visiting         
   LEAST SIMILAR: 
      -0.2365 | oil              
      -0.1732 | acquired        
      -0.1283 | dhabi            
------------------------------

WORD: **COMPUTER**
   TOP SIMILAR: 
      0.5406 | animator         
      0.5358 | cartoons         
      0.5270 | choices          
      0.4941 | device           
      0.4888 | software         
   LEAST SIMILAR: 
      -0.1905 | norway          
      -0.1289 | limestone       
      -0.1086 | argentina       
------------------------------

WORD: **SCIENCE**
   TOP SIMILAR: 
      0.5712 | agronomy         
      0.5430 | hermetic         
      0.5361 | poetical         
      0.5154 | biotechnology    
      0.5128 | occupations      
   LEAST SIMILAR: 
      -0.1101 | rising          
      -0.1058 | second          
      -0.0714 | absence         
------------------------------

WORD: **AMERICAN**
   TOP SIMILAR: 
      0.6694 | samoa            
      0.5324 | amerika          
      0.5134 | sponsored        
      0.5110 | samoans          
      0.5016 | agencies         
   LEAST SIMILAR: 
      -0.1383 | pieces          
      -0.1260 | convince        
      -0.1185 | electron        
------------------------------
