# MyGNNotes

## Neural Message Passing

1. A graph contains nodes and edges. Nodes have embeddings -- features and values of the features.  
2. Graphs can't be plotted in co-ordinate space for classification by drawing decision boundary.
3. Our goal is to ***assign meaningful co-ordinates*** to the nodes of a graph so that we can then ***easily*** create decision boundaries (rather than having to draw complicated boundaries) across the nodes for downstream classification.
    1. We want to compute ***neighbourhood-aware embeddings***  i.e., the embeddings of a node reflect the neighbourhood of the node.
    2. We use the ***message passing framework*** to achieve this goal.
5. ***Message Passing Framework:*** 
       1. Messages are the embeddings from the neighbouring nodes.
       2. Messages from all the neighbouring nodes are aggregated to compute a new embedding.
       3. The steps of message passing and aggregation can be iterated multiple times to update this new embedding.
