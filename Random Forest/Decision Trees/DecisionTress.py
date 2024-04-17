# entropy
#splits
#information_gain
#stopping criteria 
#recursive tree building
#predict
#traverse tree
#nodes
#fit

import numpy as np
from collections import Counter

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    

class DTR:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None



    def fit(self,x,y):
        self.n_features = x.shape[1] if not self.n_features else min(self.n_features,x.shape[1])
        self.root = self._grow_tree(x,y)

    def _grow_tree(self,x,y, depth=0):
        
        n_samples,n_feats = x.shape
        n_labels = len(np.unique(y))
        #check stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats,self.n_features,replace=False)

        #find best split
        best_features, best_thresh = self._best_split(x,y,feat_idxs)

        # create child nodes
        left_idx,right_idx = self._split(x[:,best_features],best_thresh)
        left = self._grow_tree(x[left_idx],y[left_idx],depth+1)
        right = self._grow_tree(x[right_idx],y[right_idx],depth+1)
        return Node(best_features,best_thresh,left,right)

    def _most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _best_split(self,x,y,feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            feature = x[:,feat_idx]
            thresholds = np.unique(feature)

            for threshold in thresholds:
                gain = self._information_gain(y,feature,threshold)
                if gain>best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx,split_thresh


    def _information_gain(self,y,x,threshold):
        parent_entropy = self._entropy(y)
        left_idx,right_idx = self._split(x,threshold)
        if len(left_idx)==0 or len(right_idx)==0:
            return 0
        n = len(y)
        n_l,n_r = len(left_idx),len(right_idx)
        e_l,e_r = self._entropy(y[left_idx]),self._entropy(y[right_idx])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        ig = parent_entropy - child_entropy
        return ig



    def _entropy(self,y):
        hist = np.bincount(y)
        ps = hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])
    
    def _split(self,x,threshold):
        left_idx = np.argwhere(x<=threshold).flatten()
        right_idx = np.argwhere(x>threshold).flatten()
        return left_idx,right_idx

    def predict(self,x):
        return np.array([self._traverse_tree(xi,self.root) for xi in x])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)