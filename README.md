
# Table of Contents

1.  [Current Project Focus](#org99f0697)
    1.  [Application to CHRONOS](#org3fa77d7)
    2.  [Why Query Focused Summarization?](#org97d77ff)
2.  [Agenda](#org31ee18d)
3.  [Current Approach](#orge04a449)
4.  [Datasets](#orge923703)
5.  [Open Ended Questions](#orgedaf6d1)
6.  [Some Ideas](#org976df43)
    1.  [Idea 1](#org93733ae)



<a id="org99f0697"></a>

# Current Project Focus

We want to develop techniques for generating *query focused summaries* of large
bodies of text using graph neural networks. Current techniques for summarization
focus on selecting sentences that are generally representative of the high level
focus of a text, however these methods do not allow for a user to specify
specific information they want to retain in the summary. We want to develop
techniques that allow for external constraints, which come in the form of
queries, to be incorporated into summarization algorithms. Additionally, we
believe in the expressive power of graphs so we want to represent text using
graphs, and operate on them using Graph Neural Networks.


<a id="org3fa77d7"></a>

### Application to CHRONOS

Currently in the CHRONOS system we generate very detailed and complicated graphs
of events that contain orders of magnitude more information than gold standard
graphs. This limits the human interpretability of these graphs and the ability
to process them for downstream tasks. We want to solve this problem using graph
summarization and pruning techniques (mentioned above). However, in order to
compare our techniques to existing forms of graph pruning and summarization we
frame our problem as summarization of natural language.


<a id="org97d77ff"></a>

### Why Query Focused Summarization?

We anticipate that generic summarization algorithms might prune important information for
downstream tasks like schema induction or template matching. For example,
specific details like dates of events and specific locations might be very
important for matching an event template like a terrorist attack, but this type of
information might be pruned from the event graph because it does not generally
summarize the document. We attempt to solve this problem by incorporating additional constraints into the
summarization process in the form of queries.


<a id="org31ee18d"></a>

# Agenda

-   [ ] Figure out good benchmark datasets
-   [ ] Setup a good baseline model
    -   [ ] Get this model setup in a Github repository
    -   [ ] Find a good baseline approach


<a id="orge04a449"></a>

# Current Approach

A current baseline approach we are thinking of using is from [Heterogeneous Graph Neural Networks for Extractive
Document Summarization](https://arxiv.org/pdf/2004.12393.pdf). They represent one or more documents as a
hierarchical graph with documents as root nodes, then sentence nodes, which are
then connected to word nodes. They use [Graph Attention Networks](https://arxiv.org/abs/1710.10903) as the primary
graph neural network layer to learn representations of each of the nodes. They
follow a stack of Graph Attention Layers with a sentence selection mechanism
based on trigram blocking ([A Deep Reinforced Model for Abstractive
Summarization](https://arxiv.org/abs/1705.04304)). They note that they don&rsquo;t use large scale language models like
[BERT](https://arxiv.org/abs/1810.04805) and reserve that for future work (they likely already have a paper coming
out that does this). This seems like a good baseline approach.

This approach does not touch on query-focused summarization, and this may be a
place we can add value.


<a id="orge923703"></a>

# Datasets

-   Query Focused Summarization Datasets
    -   [DUC 2005-2007 QFS benchmarks](https://www-nlpir.nist.gov/projects/duc/data.html)
    -   [Topically Diverse Query Focus Summarization (TD-QFS)](https://www.cs.bgu.ac.il/~talbau/TD-QFS/dataset.html)
-   General Summarization Datasets
    -   [CNN/Daily Mail](https://www.aclweb.org/anthology/K16-1028.pdf)


<a id="orgedaf6d1"></a>

# Open Ended Questions

-   How do we want to represent queries?
-   How do we want to predict relevance of each of the parts of a document to queries?
-   Do we want to extend our approach to multi-document summarization?
-   How do we want to minimize redundancy in the summary?
-   How do we want to predict how &ldquo;central&rdquo; a sentence is to summarizing a document?
-   How do we want to balance the relevance of a sentence to a query and its
    centrality to a document?
-   How can we ensure information correctness?
-   How do we want to ensure the fluency of the text in the summary?
-   How do we want to encourage efficiency of the summary sentences?
-   How do we want to represent a document (or multiple documents)?
-   What datasets and tasks do we want to evaluate our system on?
-   Can we leverage specific flexibility and information from graph data?
    structures to achieve better performance at any of the above tasks?


<a id="org976df43"></a>

# Some Ideas


<a id="org93733ae"></a>

## Idea 1

Approaches to document summarization represent documents as fully
connected graphs, where the edges represent the subjective similarity
between sentences in the document. They use similarity to model the
redundancy of the sentences and select a set of sentences that are
minimally redundant w.r.t each-other. In query focused summarization
methods model the relevance of each sentence to a query. These approaches
balance between selecting sentences that are not redundant and choosing
sentences relevant to a query. I have not seen any work that models the
redundancy of query specific information. Two sentences may have
information that pertains to a query, but we don&rsquo;t want to include the
same information multiple times in a summary. We could use graphical
representations of documents to model the redundancy of query relevance.

