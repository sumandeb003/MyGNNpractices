# MyGNNotes

## Neural Message Passing

1. A graph contains nodes and edges. Nodes have embeddings -- features and values of the features.  
2. Graphs can't be plotted in co-ordinate space for classification by drawing decision boundary.
3. Our goal is to ***assign meaningful co-ordinates*** to the nodes of a graph so that we can then ***easily*** create decision boundaries (rather than having to draw complicated boundaries) across the nodes for downstream classification.
    1. We want to compute ***neighbourhood-aware embeddings***  i.e., the embeddings of a node reflect the neighbourhood of the node.
    2. We use the ***message passing framework*** to achieve this goal.
4. ***Message Passing Framework:***
    1. Messages are the embeddings from the neighbouring nodes.
    2. Messages from all the neighbouring nodes are aggregated to compute a new embedding.
    3. The steps of message passing and aggregation can be iterated multiple times to update this new embedding.

5. GNN tasks:
    1. **Node-level task**: Regression of an attribute of a node or predict the class of a node in the given graph
    2. **Edge-level task**: Infer the existence of an edge between two existing nodes in an incomplete graph or regression/prediction of an attribute of an edge in a graph 
    3. **Graph-level task**: Classify a graph or regression/clustering task over an entire graph. The model learns to classify graphs using three main steps:
           1. Embed nodes using several rounds of message passing.
           2. Aggregate these node embeddings into a single graph embedding (called readout layer). In the code below, the average of node embeddings is used (global mean pool).
           3. Train a classifier based on graph embeddings.
            
           
![Screenshot (380)](https://user-images.githubusercontent.com/114074746/226182024-32760c06-f35d-4749-a77c-ad3524dfbb53.png)


7. GNNs are trained with batches of graphs instead of individual graphs. This is done as follows:
    1. Stack adjacency matrices in a diagonal manner leading to a large graph with multiple isolated subgraphs.
    2. Concatenate node features and the target.

![Screenshot (378)](https://user-images.githubusercontent.com/114074746/226179142-451948ae-372d-4ff5-aeae-edab15e923ac.png)

<figcaption> 

Fig: Mini-batching of graphs.

</figcaption>

8. How can edge features be used when training the model? If we take the example of GCN, it can easily be done by replacing the zeros and ones of the adjacency matrix with the edge weights.

![Screenshot (382)](https://user-images.githubusercontent.com/114074746/226182186-9a84e435-0636-442e-9c3a-1fc8efbec6ec.png)

9. Explainability of GNNs:
    i. Getting good performance is one thing, but having confidence in the prediction to take action is another. To trust the prediction of a model, one can examine the reasons why the model generated it. Sometimes, these explanations can be more important than the results themselves as they reveal the hidden patterns that the model has detected and better guide the decision-making. For graphs, explicability is about three questions: **1) Which nodes and features were relevant to making the prediction? 2) How relevant were they? 3) How relevant were the node and edge features of the graph?**

First, it is important to distinguish between:

Instance-level methods, that provide explanations at the level of individual predictions
Model-level approaches, that give explanations at the level of the whole model
Let’s explore explanations at the instance level:

Gradient or features-based methods: They rely on the gradients or hidden feature maps to approximate input importance. Gradients-based approaches compute the gradients of target prediction with respect to input features by back-propagation whereas feature-based methods map the hidden features to the input space via interpolation to measure importance scores. In this context, larger gradients or feature values mean higher importance.
Perturbation-based methods: They examine the variation in the model predictions with respect to different input perturbations. This is done by masking nodes or edges and observing the results for instance. Intuitively, predictions remain the same when important input information is kept.
Decomposition methods: They decompose prediction into the input space. Layer by layer the output is transferred back until the input layer is reached. The values then indicate which of the inputs had the highest importance on the outputs.
Surrogate: Train a simple and interpretable surrogate model to approximate the predictions of the model in the neighboring area of the input.
