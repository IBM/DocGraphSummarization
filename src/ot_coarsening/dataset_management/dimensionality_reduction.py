"""
The pre-trained models produce embeddings of size 512 - 1024. However, when storing a large
number of embeddings, this requires quite a lot of memory / storage.

In this example, we reduce the dimensionality of the embeddings to e.g. 128 dimensions. This significantly
reduces the required memory / storage while maintaining nearly the same performance.

For dimensionality reduction, we compute embeddings for a large set of (representative) sentence. Then,
we use PCA to find e.g. 128 principle components of our vector space. This allows us to maintain
us much information as possible with only 128 dimensions.

PCA gives us a matrix that down-projects vectors to 128 dimensions. We use this matrix
and extend our original SentenceTransformer model with this linear downproject. Hence,
the new SentenceTransformer model will produce directly embeddings with 128 dimensions
without further changes needed.
"""
from sklearn.decomposition import PCA
import os
import numpy as np
import torch
from abc import abstractmethod

"""
    Dimensionality reducer object that handles reducing the dimensionality
    of word and sentence embeddings
"""
class DimensionalityReducer():

    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def reduce_word_embeddings(self, word_embeddings):
        pass
    
    @abstractmethod
    def reduce_sentence_embeddings(self, sentence_embeddings):
        pass

class PCADimensionalityReducer(DimensionalityReducer):
        
    def __init__(self, dataset=None, pretrained=False, reduced_size=128, input_size=768):
        self.dataset = dataset
        self.reduced_size = reduced_size
        self.input_size = input_size
        self.batch_input_size = 2000
        self.save_path = os.path.join(os.environ["GRAPH_SUM"], "src/ot_coarsening/model_cache", "pca_dimensionality_reducer.pt")
        if pretrained:
            self.load()

    def train(self):
        # go through and save the x values for a batch from the dataset
        print("training")
        x_values = []
        for i in range(self.batch_input_size):
            graph = self.dataset.get(i, reduce_dimensionality=False)
            x_values.append(graph.x)
        x_values = torch.cat(x_values, dim=0)
        x_values = x_values.detach().cpu().numpy()
        # generate a PCA matrix for those values
        pca = PCA(n_components=self.reduced_size)
        pca.fit(x_values)
        self.pca = pca
    
    def save(self):
        print("saving")
        # serialize this object with torch
        torch.save(self.pca, self.save_path)

    def load(self):
        # load a pretrained pca object 
        self.pca = torch.load(self.save_path)

    def reduce(self, embeddings, device=None):
        # convert the embeddings to numpy
        if device is None:
            embeddings_device = torch.device("cpu")
        else:
            embeddings_device = device
        embeddings = embeddings.detach().cpu().numpy()
        reduced_embeddings = self.pca.transform(embeddings)
        reduced_embeddings = torch.Tensor(reduced_embeddings).to(embeddings_device)
        return reduced_embeddings

    def reduce_word_embeddings(self, word_embeddings):
        # TODO  make independent reduction systems for word embeddings and sentence embeddings
        pass

    def reduce_sentence_embeddings(self, sentence_embeddings):
        # TODO  make independent reduction systems for word embeddings and sentence embeddings
        pass


