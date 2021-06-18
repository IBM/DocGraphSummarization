import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Dataset, Data
import torch_geometric.transforms as T
import jsonlines
import psutil
import os
from tqdm import tqdm
GRAPH_SUM = os.environ["GRAPH_SUM"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_memory_usage():
    # gives a single float value
    psutil.cpu_percent()
    # gives an object with many fields
    psutil.virtual_memory()
    # you can convert that object to a dictionary 
    dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    percentage = psutil.virtual_memory().percent
    # print("Virtual memory percentage {}".format(percentage))

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

"""
    CNNDailyMail Dataset
    - It is in memory because the dataset is fairly small
    - I referenced this page https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    - The data that I used for this class was preprocessed by the HeterSumGraph people
    - The data is located at GraphSummarization/data/CNNDM
    - For future datasets I will need to do the TF-IDF preprocessing and such on my own
    - I am doing an in memory dataset (the train is 5GB), but it loads into CPU memory so that should be fine. I may need to change that later
"""
class CNNDailyMail(Dataset):
    def __init__(self, transform=None, pre_transform=None, mode="train", graph_constructor=None, perform_processing=False, proportion_of_dataset=1.0, max_number_of_nodes=1000):
        self.root = os.path.join(GRAPH_SUM, "data/CNNDM")
        self.mode = mode # "trian", "test", or "val"
        self.graph_constructor = graph_constructor
        self.proportion_of_dataset = proportion_of_dataset
        self.perform_processing = perform_processing
        self.max_number_of_nodes = max_number_of_nodes
        # self.data, self.slices = torch.load(self.processed_paths[0])
        super(CNNDailyMail, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # files = [tfidfs, labels, vocab]
        if self.mode == "train":
            return [
                "unprocessed/train.w2s.tfidf.jsonl",
                "unprocessed/train.label.jsonl",
                "unprocessed/vocab",
            ]
        elif self.mode == "test":
            return [
                "unprocessed/test.w2s.tfidf.jsonl",
                "unprocessed/test.label.jsonl",
                "unprocessed/vocab",
            ]
        elif self.mode == "val":
            return [
                "unprocessed/val.w2s.tfidf.jsonl",
                "unprocessed/val.label.jsonl",
                "unprocessed/vocab",
            ]
        else:
            raise Exception("Unrecognized mode: {}".format(self.mode))

    @property
    def processed_file_names(self):
        return os.listdir(os.path.join(self.root, "processed", self.mode)) # ['processed/train.pt']
        """
                if self.mode == "train":
                    return os.listdir(os.path.join(self.root, self.mode)) # ['processed/train.pt']
                elif self.mode == "test":
                    return ['processed/test.pt']
                elif self.mode == "val":
                    return ['processed/val.pt']
                else:
                    raise Exception("Unrecognized mode: {}".format(self.mode))
        """
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
        return len(os.listdir(os.path.join(self.root, "processed", self.mode)))

    """
        Reads the given raw files into json data objects
    """
    def _read_raw_files_to_data(self, tfidf_file, label_file, vocab):
        print("Reading raw files to data ...")
        stopping_index = file_len(os.path.join(self.root, tfidf_file)) * self.proportion_of_dataset
        current_index = 0
        with jsonlines.open(os.path.join(self.root, tfidf_file)) as tfidf_reader:
            with jsonlines.open(os.path.join(self.root, label_file)) as label_reader:
                for index, tfidf in enumerate(tqdm(tfidf_reader)):
                    # check number of nodes
                    label = label_reader.read()
                    get_memory_usage()
                    data = self._construct_graph(tfidf, label)
                    number_of_nodes = data.x.shape[0] 
                    if number_of_nodes > self.max_number_of_nodes:
                        continue
                    # save the data
                    save_path = os.path.join(self.root, "processed", self.mode, "data_{}.pt".format(index))
                    torch.save(data, save_path)
                    current_index += 1
                    if current_index > stopping_index:
                        break

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
        tfidf_file, label_file, vocab = raw_file_names
        data_list = self._read_raw_files_to_data(tfidf_file, label_file, vocab)
        # if self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]
        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]
    
    @property
    def num_node_features(self):
        return self.graph_constructor.word_embedding_size

    def get(self, idx):
        data_dir = os.path.join(self.root, "processed", self.mode)
        graph_data_path = os.path.join(data_dir, "data_{}.pt".format(idx))
        data = torch.load(graph_data_path)
        data = T.ToDense(self.max_number_of_nodes)(data)
        data.adj = data.adj.squeeze().float()
        # pad y
        num_to_pad = 3 - data.y.shape[0]
        data.y = F.pad(input=data.y, pad=(0, num_to_pad), mode='constant', value=-1)
        data = data.to(device)
        return data
