# TDGNN：Continuous-Time Link Prediction via Temporal Dependent Graph Neural Network
This is our Pytorch implementation for the paper:

> Liang Qu, Huaisheng Zhu, Qiqi Duan, and Yuhui Shi. 2020. Continuous-Time Link Prediction via Temporal Dependent Graph Neural Network. In Proceedings of The Web Conference 2020 (WWW '20). Association for Computing Machinery, New York, NY, USA, 3026–3032. DOI:https://doi.org/10.1145/3366423.3380073

## Introduction

Temporal Dependent Graph Neural Network (TDGNN), a simple yet effective dynamic network representation learning framework which incorporates the network temporal information into GNNs. TDGNN introduces a novel Temporal Aggregator (TDAgg) to aggregate the neighbor nodes’ features and edges’ temporal information to obtain the target node representations.

## Citation

If you want to use our codes in your research, please cite:

``` 
@inproceedings{10.1145/3366423.3380073,
author = {Qu, Liang and Zhu, Huaisheng and Duan, Qiqi and Shi, Yuhui},
title = {Continuous-Time Link Prediction via Temporal Dependent Graph Neural Network},
year = {2020},
isbn = {9781450370233},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3366423.3380073},
doi = {10.1145/3366423.3380073},
booktitle = {Proceedings of The Web Conference 2020},
pages = {3026–3032},
numpages = {7},
location = {Taipei, Taiwan},
series = {WWW '20}
}
```

## Usage

``` 
python3 model.py -input_node ../contact/feature_random_contact.txt -input_edge_train ../contact/edge_train_contact -input_edge_test ../contact/edge_train_contact -output_file result -aggregate_function origin -hidden_dimension 128
```



