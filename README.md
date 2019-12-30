A Toolkit for Neural Review-based Recommendation models with Pytorch.
基于评论文本的深度推荐系统模型库 (Pytorch)

# Neural Review-based Rec
In this repository, we reimplement some important review-based recommendation models, and provide an extensible framework **NRRec**  with Pytorch.
Researchers can implement their own methodss easily in our framework (just in *models* folder).


## Introduction to Review-based Recommendaiton

E-commerce platforms allow users to post their reviews towards products, and the reviews may contain the opinions of users and the features of the items.
Hence, many works start to utilize the reviews to model user preference and item features.
Traditional methods always use topic modelling technology to capture  the semantic informtion.
Recently, many deep learning based methods are proposed, such as DeepCoNN, D-Attn etc, which use the neural networks, attention mechanism to learn representations of users and items more comprehensively.


More details please refer to my blog (updating).

## Methods

>Note: since each user and each item would have multiple reviews, we categorize the existing methods into two kinds:
- document-level methods: concatenate all the reviews into a long document, and then learn representations from the doc, we denote as **Doc** feature.
- review-level methods: model each review seperately and then aggregate all reviews together as the user item latent feature.
Besides, the rating feature of users and items (i.e., ID embedding) is usefule when there are few reviews.

We plan to follow the start-of-art review-based recommendation methods and involve them into this repo, the baseline methods are listed here:

| Method | Feature | Prediction | Status|
| ---- | ---- | ---- | ---- |
| DeepCoNN | Doc| FM | &check; |
| DAML | Doc, ID |  | &check; |
|NARRE | Review, ID | LFM | &check; |
| D-Atten | Doc | MLP | &#9746;|
| ANR | Doc, ID |  | &#9746;|
| CARL | Doc, ID |  | &#9746;|
| CARP | Doc, ID | | &#9746;|
|MPCN | Review, ID | FM |&#9746; |
|TARMF | Review, ID | FM | &#9746;|

We will release the rest baseline methods later.

## Usage

**Requirements**

- Python >= 3.6
- Pytorch >= 1.0
- fire: commend line parameters (in `config/config.py`)
- numpy, gensim etc.


**Use the code**

- Preprocessing the origin Amazon or Yelp dataset via `pro_data/data_pro.py`, then some `npy` files will be generated in `dataset/`
    ```
    cd pro_data
    python3 data_pro.py Digital_Music_5.json
    # details in data_pro.py (e.g., the pretrained word2vec.bin path)
    ```
- Run the model. Take DeepCoNN and NARRE as examples, the command lines can be customized:
    ```
    python3 main.py train --model=DeepCoNN --num_fea=1 --output=fm
    python3 main.py train --model=NARRE --num_fea=2 --output=lfm
    ```
    Note that the `num_fea (1,2,3)` corresponds how many features used in the methods, (ID feature, Review and Doc)

## Framework Design

An overview of the package dir:
![framework](http://cdn.htliu.cn/blog/review-based-rec/code.png)

### Data Preprocessing
After data processing, one record of the training/validation/test dataset is:
```
user_id, item_id, ratings
```
For example the training data triples are stored as `Train.npy, Train_Score.npy` in `dataset/Digital_Music_data/train/`.

The review information of users and items are preprocessed in the following format:

- user_id
- user_doc: the word index sequence of the document of the user, `[w1, w2, ... wn]`
- user_reviews list: the list of all the review of the user, `[[w1,w2..], [w1,w2,..],...[w1,w2..]]`
- user_item2id: the item ids that the user have purchased, `[id1, id2,...]`

The same as the items. Hence in the code, we orgnize our batch data as:
```
uids, iids, user_reviews, item_reviews, user_item2id, item_user2id, user_doc, item_doc
```
This is all the information involved in review-based recommendation, researchers can utilize this data format to build own models.
Note that the review in validation/test dataset is excluded.

### Model Details
In order to make our framework more extensible, we define three modules in our framework:

- User/Item Representation Learning Layer (in `models/*py`): the main part of most baseline methods, such as the CNN encoder in DeepCoNN.
- Fusion Layer in `framework/fusion.py`: combine the user/item different features (e.g., ID feature and review/doc feature), and then fuse the user and item feature into one feature vector, we pre-define the following methods:
    - sum
    - add
    - concatenation
    - self attention
- Prediction Layer in `framework/prediction.py`: prediction the score that user towards item (i.e., a regression layer), we pre-define the following rating prediction layers:
    - (Neural) Factorization Machine
    - Latent Factor Model
    - MLP

Hence, researchers could build their models in user/item representation learning layer.

