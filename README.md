## 1 Identifying Idioms with Transformers Architectures

In the field of Natural Language Processing, accurately and efficiently identifying idiomatic expressions in French presents a significant challenge. Our project focuses on this task, utilizing an architecture composed of a transformer to capture word embeddings followed by a single-layer perceptron to classify words according to a specific output encoding.

The main objective is to determine which output encoding - BIOG (Beginning, Inside, Outside, Gap) or IOG (Inside, Outside, Gap) - performs best in detecting French expressions. These encoding methods will be examined with and without the integration of the Viterbi algorithm, a classical tool for optimizing output sequences.

## 2 Baseline

Our baseline relies on a simple approach of memorizing expressions. This method involves building a list of expressions from a training text, retaining each word group where each word is marked as part of the expression, with at most one unmarked intermediary word. For example, the expression "avoir pour but" is retained, while "avoir pour principal but" is not.

The next challenge was to compare the performances of the baseline and the model used subsequently, hence making the learned expression list usable to generate a BIOG or IOG output encoding for a given input sequence. To achieve this, we start by identifying the expressions present in the input sequence and in the learned expression list. Then, it remains to label the word based on its presence in an expression or not.

| Encoding | O | B | I | G |
|----------|---|---|---|---|
| Baseline BIOG | 95.7 | 3.9 | 0.9 | 1.9 |
| Baseline IOG | 97 | - | 3.9 | 0.9 |

Table 1 – F1 scores for BIOG and IOG encodings for the baseline.

## 3 Model

### 3.1 Transformers

The CamemBERT model is used for various Natural Language Processing tasks. CamemBERT operates by breaking sentences into words or sub-words (tokens), which are then converted into vectors using an embedding layer. These vectors are processed by 12 hidden layers, using self-attention and dense transformations. The model takes token sequences as input and processes sequences of fixed length, sometimes requiring the addition of padding tokens.

However, training CamemBERT on our computers was considerably slower and more resource-intensive compared to other models. As a result, we chose to use DistilBERT, a lighter and faster version of BERT.

### 3.2 Input Encoding

DistilBERT's tokenizer plays a crucial role in data preparation for analysis. It decomposes sentences into tokens, but not always into complete words. For example, the word "incompréhensible" could be divided into "in", "##compr", "##ehen", "##sible". This segmentation poses a problem: how to match the corpus labels to the embeddings output by DistilBERT?

To overcome this issue, we choose to average the embeddings. Taking the example above, the word "incompréhensible" will return 4 embeddings, thus 4 vectors of size 512x768. However, in a sentence containing multiple embeddings, it is possible to lose the correspondence between labels and embeddings. To address this, we average the embeddings so that a sequence of 10 words returns 10 embeddings.

This approach is essential to maintain the semantic and contextual coherence of words in the model.

### 3.3 Comparison of Output Encodings

In our project, BIOG and IOG encodings play a crucial role. BIOG stands for "Beginning, Inside, Outside, Gap" and allows for a fine distinction between the different parts of an idiomatic expression. For example, in the expression "prendre son envol," "prendre" would be marked as 'B,' "son" as 'G,' and "envol" as 'I.' IOG, standing for "Inside, Outside, Gap," is a simplified version, where only words inside and outside the expression are distinguished. This choice of encoding directly impacts the model's accuracy in identifying idiomatic expressions in French.

The BIOG encoding, being more detailed, could theoretically provide a better understanding of the boundaries of each expression, while IOG, being simpler, could prove sufficient and more effective for certain applications. Our analysis will seek to determine if the additional complexity of BIOG brings an advantage over IOG and if this advantage justifies the increase in model complexity and processing time.

Finally, we also consider the potential integration of the Viterbi algorithm to optimize output sequences.

### 3.4 Viterbi Algorithm

The Viterbi algorithm is a tool that allows us to improve the accuracy of our project. It is used to determine the most probable sequence of labels (such as BIOG or IOG) for a sequence of words in a sentence.

The algorithm operates by computing transition probabilities between labels and word emission probabilities for these labels, based on training data. In practice, it allows tracing the most probable path through these probabilities, thereby maximizing precision in identifying expressions. In our project, the use of Viterbi is expected to enhance the precision and efficiency of the model.

### 3.5 Training and Results

Our model is based on the DistilBERT architecture followed by a single-layer perceptron. In our task, each word in the text is classified according to the BIOG or IOG encoding, where each letter represents a type of positioning in an expression (Beginning, Inside, Outside, Gap).

We will use the F1-score to evaluate and compare our models.

The model training takes place in several steps. We start by preprocessing the text using the DistilBERT tokenizer, which tokenizes the text. These tokens are then passed through the DistilBERT model to obtain embeddings. We then average the embeddings of each word to obtain a unique representation per word.

Our model also includes a dense neural network, which takes the DistilBERT embeddings and predicts a label for each word. The hidden layer consists of 334 neurons, which corresponds to half the number of neurons in the input layer (hyperparameter optimization could have been possible here, but we did not have the time). During training, we use a CrossEntropy loss function, weighted based on the class distribution in our dataset. We also use techniques like Early Stopping and learning rate scheduling to optimize training and avoid overfitting.

We conducted an initial experiment where we used "naive" embeddings, meaning that we directly employed the embeddings from DistilBERT without modification or averaging. Thus, a large number of words were considered unknown by the tokenizer.

The results are summarized in the following table for training over 30 epochs and a learning rate of 10^-4 (with a learning rate scheduler):

| Encoding | O | B | I | G |
|----------|---|---|---|---|
| BIOG w/o Viterbi | 90.3 | 29.7 | 25.0 | 13.8 |
| BIOG w/ Viterbi | 98.2 | 55.6 | 56.4 | 37.1 |
| IOG w/o Viterbi | 94.2 | - | 35.5 | 21.2 |
| IOG w/ Viterbi | 98.5 | - | 56.7 | 37.8 |

Table 2 – F1 scores for BIOG and IOG encodings with and without the Viterbi algorithm with "naive" embeddings.

Subsequently, we attempted to use the averaging method of embeddings to decrease the number of unknown tokens in our corpus.
## Conclusion

In conclusion, our project focuses on identifying idiomatic expressions in French using transformer architectures. We compare the performance of different output encodings and explore the integration of the Viterbi algorithm to optimize output sequences. Our experiments show promising results, especially with the use of Viterbi, indicating the potential for improved accuracy in identifying idiomatic expressions.

Moving forward, we plan to conduct further experiments to refine our model and explore additional techniques to enhance its performance. Our ultimate goal is to develop a robust system capable of accurately identifying idiomatic expressions in French, which could have various applications in natural language understanding and processing.

## References

[1] Example Reference

