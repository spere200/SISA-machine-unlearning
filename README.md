# Exact Unlearning With SISA (Work in Progress)
### Information
An implementation of exact unlearning using a SISA approach as described in [An Introduction to Machine Unlearning](https://arxiv.org/abs/2209.00939) (Mercuri et al.). The dataset used is the KDD Cup 1999 dataset which can be found [here (Kaggle)](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data).

### Motivation
The idea behind this impementation with is to create a classifier capable of forgetting datapoints. This can be useful for things like complying with data removal requests, improving model resilience by removing malicious datapoints, etc. 

The current method of dealing with these issues is naive unlearning, which simply consists of removing the datapoints from the set and retraining the model from scratch, but this can take an exceedingly long amount of time. With SISA, only one of the aggregated learning models needs to be retrained, and since learning is performed in an incremental manner, snapshots of each model are taken at each training step, which means all that needs to be done upon receiving a deletion request is identify in which step the datapoint was "learned", and redo all learning only on that model and only from that snapshot onwards, drastically reducing re-training time for the entire model.

### Implementation Details
Most of this implementation relies on numpy, pandas, and sklearn. The SISA class receives shards and slices as integer parameters, and creates as many [sklearn.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) models as there are shards, with loss="log_loss" for logistic regression. 

SGDClassifier is used due to its [partial_fit()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.partial_fit) method, which allows for incremental learning. This lets us break each shard further into slices, and we clone the model after each partial fit on a slice.