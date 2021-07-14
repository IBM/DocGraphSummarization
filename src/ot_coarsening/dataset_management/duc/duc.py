import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Dataset, Data
import torch_geometric.transforms as T
import jsonlines
import psutil
import traceback
import os
import sys
from tqdm import tqdm
GRAPH_SUM = os.environ["GRAPH_SUM"]
import sys
sys.path.append(GRAPH_SUM)
from src.ot_coarsening.dataset_management.dimensionality_reduction import PCADimensionalityReducer

device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

"""
    DUC Dataset
"""
class DUC(Dataset):
    def __init__(self, transform=None, pre_transform=None, mode="train", year="2005", graph_constructor=None, perform_processing=False, proportion_of_dataset=1.0, max_number_of_nodes=1000, overwrite_existing=False, reduce_dimensionality=False):
        self.root = "/dccstor/helbling1/data/DUC"
        self.mode = mode # "trian", "test", or "val"
        self.graph_constructor = graph_constructor
        self.dimensionality = 768
        self.reduce_dimensionality = reduce_dimensionality
        if self.reduce_dimensionality:
            self.dimensionality_reducer = PCADimensionalityReducer()
            self.dimensionality_reducer.load()
            self.dimensionality = self.dimensionality_reducer.reduced_size
        self.proportion_of_dataset = proportion_of_dataset
        self.year = year
        self.perform_processing = perform_processing
        self.overwrite_existing = overwrite_existing
        self.max_number_of_nodes = max_number_of_nodes
        self.max_summary_length = 10
        self.num_sentence_nodes = None
        # self.data, self.slices = torch.load(self.processed_paths[0])
        super(DUC, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # files = [tfidfs, labels, vocab]
        return [
            "unprocessed/DUC"+self.year+"/tfidf.jsonl",
            "unprocessed/DUC"+self.year+"/label.jsonl",
        ]

    @property
    def label_path(self):
        return [self.raw_file_names[1]]

    @property
    def processed_file_names(self):
        return os.listdir(os.path.join(self.root, "processed", "DUC"+self.year))

    def download(self):
        # do nothing becuase the dataset is downloaded
        pass

    """
        Takes in the preprocessed data and generates a graph from it in the
        pytorch geometric style. It embeds the words and sentences using variants
        of pretrained BERT and computes
    """
    def _construct_graph(self, tfidf, label):
        return self.graph_constructor.construct_graph(tfidf, label)

    def len(self):
        return int(len(os.listdir(os.path.join(self.root, "processed", self.mode))) * self.proportion_of_dataset)

    """
        Reads the given raw files into json data objects
    """
    def _read_raw_files_to_data(self, tfidf_file, label_file):
        print("Reading raw files to data ...")
        stopping_index = file_len(os.path.join(self.root, tfidf_file)) * self.proportion_of_dataset
        current_index = 0
        with jsonlines.open(os.path.join(self.root, tfidf_file)) as tfidf_reader:
            with jsonlines.open(os.path.join(self.root, label_file)) as label_reader:
                for index, tfidf in enumerate(tqdm(tfidf_reader)):
                    try: 
                        # read label
                        label = label_reader.read()
                        print("read label and tfidf")
                        # check if file exists
                        save_path = os.path.join(self.root, "processed", self.mode, "data_{}.pt".format(index))
                        if not self.overwrite_existing and os.path.exists(save_path):
                            current_index += 1
                            continue
                        # num sentences
                        # check number of nodes
                        # get_memory_usage()
                        graph = self._construct_graph(tfidf, label)
                        if graph is None:
                            continue
                        number_of_nodes = graph.x.shape[0] 
                        if number_of_nodes > self.max_number_of_nodes:
                            continue
                        # save the data
                        torch.save(graph, save_path)
                        current_index += 1
                        if current_index > stopping_index:
                            break
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print("Issue while constructing graph")
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print(str(e))
                        print(traceback.format_exc())

    """
        Setup graph dataset preprocessing
    """
    def _setup_graph_preprocessing(self):
        num_nodes = max_num_nodes = 0
        for data in self.data:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        num_nodes = min(int(num_nodes / len(self) * 5), max_num_nodes)

        indices = []
        for i, data in tqdm(enumerate(self.data)):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        self.data = self.data[torch.tensor(indices)]

        if not sparse:
            if self.transform is None:
                self.transform = T.ToDense(num_nodes)
            else:
                self.transform = T.Compose(
                    [self.transform, T.ToDense(num_nodes)])

    def process(self):
        if not self.perform_processing:
            return
        # Load the raw files
        raw_file_names = self.raw_file_names
        tfidf_file, label_file = raw_file_names
        data_list = self._read_raw_files_to_data(tfidf_file, label_file)
        # if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]
        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]
    
    @property
    def num_node_features(self):
        return self.graph_constructor.word_embedding_size

    def reshape_graph(self, example_graph):
        example_graph.adj = example_graph.adj.unsqueeze(0)
        num_nodes = example_graph.adj.shape[2]
        example_graph.adj = torch.reshape(example_graph.adj, (1, num_nodes, -1))
        example_graph.x = example_graph.x.unsqueeze(0)  
        embedding_shape = example_graph.x.shape[2]
        example_graph.x = torch.reshape(example_graph.x, (1, num_nodes, -1))
        example_graph.y = example_graph.y.unsqueeze(0) 
        example_graph.y = torch.reshape(example_graph.y, (1, -1))
        example_graph.mask = example_graph.mask.unsqueeze(0)
        example_graph.mask = torch.reshape(example_graph.mask, (1, -1))
        return example_graph

    """
        Reduces the dimensionality of the word and sentence embeddings using
        a basic dimensionality reduction technique like PCA, pretrianed on a subset 
        of the data.
    """
    def _reduce_embedding_dimensionality(self, graph, pretrained=True):
        if not pretrained:
            raise Exception("I have not implemented the non-pretrained embedding system")
        # get num sentences
        num_sentences = len(graph.label["text"])
        # unpack word and sentence embeddings
        embeddings = graph.x
        # apply the reducer on word embeddings
        new_x = self.dimensionality_reducer.reduce(embeddings, device=device)
        # alter the graph word and sentence embeddings
        graph.x = new_x
        return graph
    
    def get(self, idx, dense=False):
        data_dir = os.path.join(self.root, "processed", self.mode)
        graph_data_path = os.path.join(data_dir, "data_{}.pt".format(idx))
        # check if path exists
        if not os.path.exists(graph_data_path):
            return self.get((idx - 1) % self.len())
        graph = torch.load(graph_data_path)

        if dense:
            graph = T.ToDense(self.max_number_of_nodes)(graph)
            graph.adj = graph.adj.squeeze().float()
            graph = self.reshape_graph(graph)
        # Reduce embedding dimensionality
        if self.reduce_dimensionality:
            graph = self._reduce_embedding_dimensionality(graph, pretrained=True)
        num_to_pad = self.max_summary_length - graph.y.shape[0]
        graph.y = F.pad(input=graph.y, pad=(0, num_to_pad), mode='constant', value=-1)
        return graph

