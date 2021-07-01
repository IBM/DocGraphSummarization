from graph_constructor import CNNDailyMailGraphConstructor
from dimensionality_reduction import PCADimensionalityReducer
from cnn_daily_mail import CNNDailyMail

if __name__ == "__main__":
    # TODO make sure the system uses the same word and sentence transformer
    # train a dimensionality reducer
    graph_constructor = CNNDailyMailGraphConstructor()
    dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=False)
    pca_reducer = PCADimensionalityReducer(dataset)
    pca_reducer.train()
    pca_reducer.save()
