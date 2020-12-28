<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
TDGNN
-----
A Temporal Dependent Graph Neural Network
(TDGNN), a simple yet effective dynamic network representation
learning framework which incorporates the network temporal information
into GNNs. 
   
Usage
----
**-input_node**: *input_node_filename*, e.g.  
Each line in the file contains the following information:
       
    0 0.1 0.2 0.3 0.4 ...  
The first element is node index and others are representation of this node  

**-input_edge_train**: *input_edge_train_filename*, e.g.  
Each line in the file contains the following information:

    0 1
The first and second elements are nodes and this represents there is an edge between these two nodes   
**-input_edge_test**: *input_edge_test_filename*, e.g.  
Each line in the file contains the following information:

    0 1
The first and second elements are nodes and this represents there is an edge between these two nodes   
**-output_file**: *output_filename*, this file is used to store the result of each epoch  
**-aggregate_function**: *aggregate_function*, there are only five functions to choose (*mean*,*had*,*w1*,*w2*,*origin*)   
**-hidden_dimension**: *hidden_dimension*, Dimensions for hidden layer for GNN 

Implementation
----
Here, we will show how to implement TDGNN on the *contact* dataset used in the TDGNN paper.  
Example Command:

    python3 model.py -input_node ../contact/feature_random_contact.txt -input_edge_train ../contact/edge_train_contact -input_edge_test ../contact/edge_train_contact -output_file result -aggregate_function origin -hidden_dimension 128
