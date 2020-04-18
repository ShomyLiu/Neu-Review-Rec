# -*- coding: utf-8 -*-
import json
import pandas as pd
import re
import sys
import os
import numpy as np
import time
from sklearn.model_selection import train_test_split
from operator import itemgetter
import gensim
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

P_REVIEW = 0.85
MAX_DF = 0.7
MAX_VOCAB = 40000
DOC_LEN = 400
PRE_W2V_BIN_PATH = ""


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def get_count(data, id):
    # count,每个id参与评论的次数
    idList = data[[id, 'ratings']].groupby(id, as_index=False)
    idListCount = idList.size()
    return idListCount


def numerize(data):
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


def bulid_vocbulary(xDict):
    rawReviews = []
    for (id, text) in xDict.items():
        rawReviews.append(' '.join(text))
    return rawReviews


def build_doc(u_reviews_dict, i_reviews_dict):
    '''
    https://github.com/cartopy/ConvMF/blob/master/data_manager.py#L432
    build doc from reviews and remove some words
    '''
    u_reviews = []
    for ind in range(len(u_reviews_dict)):
        u_reviews.append(' '.join(u_reviews_dict[ind]))

    i_reviews = []
    for ind in range(len(i_reviews_dict)):
        i_reviews.append(' '.join(i_reviews_dict[ind]))

    vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB, stop_words='english')
    vectorizer.fit(u_reviews)
    vocab = vectorizer.vocabulary_

    def clean_review(rDict):
        new_dict = {}
        for k, text in rDict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def clean_doc(raw):
        new_raw = []
        for line in raw:
            review = [word for word in line.split() if word in vocab]
            if len(review) > DOC_LEN:
                review = review[:DOC_LEN]
            new_raw.append(review)
        return new_raw

    u_reviews_dict = clean_review(user_reviews_dict)
    i_reviews_dict = clean_review(item_reviews_dict)

    u_doc = clean_doc(u_reviews)
    i_doc = clean_doc(i_reviews)

    return vocab, u_doc, i_doc, u_reviews_dict, i_reviews_dict


def countNum(xDict):
    minNum = 100
    maxNum = 0
    sumNum = 0
    maxSent = 0
    minSent = 3000
    # pSentLen = 0
    ReviewLenList = []
    SentLenList = []
    for (i, text) in xDict.items():
        sumNum = sumNum + len(text)
        if len(text) < minNum:
            minNum = len(text)
        if len(text) > maxNum:
            maxNum = len(text)
        ReviewLenList.append(len(text))
        for sent in text:
            # SentLenList.append(len(sent))
            if sent != "":
                wordTokens = sent.split()
            if len(wordTokens) > maxSent:
                maxSent = len(wordTokens)
            if len(wordTokens) < minSent:
                minSent = len(wordTokens)
            SentLenList.append(len(wordTokens))
    averageNum = sumNum // (len(xDict))

    # #################以85%的覆盖率确定句子最大长度##########################
    # 将所有review的长度按照从小到大排序
    x = np.sort(SentLenList)
    # 统计有多少个评论
    xLen = len(x)
    # 以p覆盖率确定句子长度
    pSentLen = x[int(P_REVIEW * xLen) - 1]
    x = np.sort(ReviewLenList)
    # 统计有多少个评论
    xLen = len(x)
    # 以p覆盖率确定句子长度
    pReviewLen = x[int(P_REVIEW * xLen) - 1]

    return minNum, maxNum, averageNum, maxSent, minSent, pReviewLen, pSentLen


if __name__ == '__main__':

    assert(len(sys.argv) >= 2)
    filename = sys.argv[1]

    yelp_data = False
    if len(sys.argv) > 2 and sys.argv[2] == 'yelp':
        yelp_data = True

    save_folder = '../dataset/' + filename[:-7]+"_data"

    if len(sys.argv) == 3:
        save_folder = '../dataset/' + filename[:-3]+"_data"
    print("数据集名称：{}".format(save_folder))

    if not os.path.exists(save_folder + '/train'):
        os.makedirs(save_folder + '/train')
    if not os.path.exists(save_folder + '/test'):
        os.makedirs(save_folder + '/test')

    file = open(filename, errors='ignore')

    users_id = []
    items_id = []
    ratings = []
    reviews = []

    # --------------------------------------for yelp16----------------------------------------------
    if yelp_data:
        for line in file:
            value = line.split('\t')
            reviews.append(value[2])
            users_id.append(value[0])
            items_id.append(value[1])
            ratings.append(value[3])

    # ---------------------------------for amazon --------------------------------------------------
    else:
        for line in file:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['asin']) == 'unknown':
                print("unkown item id")
                continue
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            ratings.append(str(js['overall']))

    data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
                  'ratings': pd.Series(ratings), 'reviews': pd.Series(reviews)}
    data = pd.DataFrame(data_frame)[['user_id', 'item_id', 'ratings', 'reviews']]
    # ================释放内存============#
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    # ====================================#
    userCount, itemCount = get_count(data, 'user_id'), get_count(data, 'item_id')
    userNum_raw = userCount.shape[0]
    itemNum_raw = itemCount.shape[0]
    print("===============Start: rawData size======================")
    print("dataNum: {}".format(data.shape[0]))
    print("userNum: {}".format(userNum_raw))
    print("itemNum: {}".format(itemNum_raw))
    print("data densiy: {:.5f}".format(data.shape[0]/float(userNum_raw * itemNum_raw)))
    print("===============End: rawData size========================")
    uidList = userCount.index  # userID列表
    iidList = itemCount.index  # itemID列表
    user2id = dict((uid, i) for(i, uid) in enumerate(uidList))
    item2id = dict((iid, i) for(i, iid) in enumerate(iidList))
    data = numerize(data)

    # ########################在构建字典库之前，先划分数据集###############################
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=1234)
    # 重新统计训练集中的用户数，商品数，查看是否有丢失的数据
    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uidList_train = userCount.index
    iidList_train = itemCount.index
    userNum = userCount.shape[0]
    itemNum = itemCount.shape[0]
    print("===============Start-no-process: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum: {}".format(userNum))
    print("itemNum: {}".format(itemNum))
    print("===============End-no-process: trainData size========================")

    iidMiss = []
    if itemNum != itemNum_raw:
        for iid in range(itemNum_raw):
            if iid not in iidList_train:
                iidMiss.append(iid)

    iid_concat_index = []
    for iid in iidMiss:
        index = data_test.index[data_test['item_id'] == iid].tolist()[0]
        iid_concat_index.append(index)
        
    data_train = pd.concat([data_train, data_test.iloc[iid_concat_index]])

    uidMiss = []
    if userNum != userNum_raw or itemNum != itemNum_raw:
        for uid in range(userNum_raw):
            if uid not in uidList_train:
                uidMiss.append(uid)

    uid_concat_index = []
    for i in range(len(uidMiss)):
        index = data_test.index[data_test['user_id'] == uidMiss[i]].tolist()[0]
        uid_concat_index.append(index)

    data_train = pd.concat([data_train, data_test.iloc[uid_concat_index]])

    concat_index = list(set().union(uid_concat_index,iid_concat_index))
    data_test = data_test.drop(data_test.index[concat_index])

    data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)
    # 重新统计训练集中的用户数，商品数，查看是否有丢失的数据
    userCount, itemCount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    uidList_train = userCount.index
    iidList_train = itemCount.index
    userNum = userCount.shape[0]
    itemNum = itemCount.shape[0]
    print("===============Start-already-process: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum: {}".format(userNum))
    print("itemNum: {}".format(itemNum))
    print("===============End-already-process: trainData size========================")

    user_nodes = []
    item_nodes = []
    x_train = []
    y_train = []
    for i in data_train.values:
        uiList = []
        uid = i[0]
        iid = i[1]
        score = i[2]
        user_nodes.append(int(uid))
        item_nodes.append(int(iid))
        uiList.append(uid)
        uiList.append(iid)
        x_train.append(uiList)
        y_train.append(float(i[2]))

    x_val = []
    y_val = []
    for i in data_test.values:
        uiList = []
        uid = i[0]
        iid = i[1]
        uiList.append(uid)
        uiList.append(iid)
        x_val.append(uiList)
        y_val.append(float(i[2]))

    np.save("{}/train/Train.npy".format(save_folder), x_train)
    np.save("{}/train/Train_Score.npy".format(save_folder), y_train)
    np.save("{}/test/Test.npy".format(save_folder), x_val)
    np.save("{}/test/Test_Score.npy".format(save_folder), y_val)

    print("{} 测试集大小{}".format(now(), len(x_val)))
    print("{} 测试集评分大小{}".format(now(), len(y_val)))
    print("{} 训练集大小{}".format(now(), len(x_train)))

    # #####################################2，构建字典库，只针对训练数据############################################
    user_reviews_dict = {}
    item_reviews_dict = {}
    # 新增项
    user_iid_dict = {}
    item_uid_dict = {}
    user_len = defaultdict(int)
    item_len = defaultdict(int)

    for i in data_train.values:
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))

        if len(str_review.strip()) == 0:
            str_review = "<unk>"

        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(str_review)
            user_iid_dict[i[0]].append(i[1])
        else:
            user_reviews_dict[i[0]] = [str_review]
            user_iid_dict[i[0]] = [i[1]]

        if item_reviews_dict.__contains__(i[1]):
            item_reviews_dict[i[1]].append(str_review)
            item_uid_dict[i[1]].append(i[0])
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_uid_dict[i[1]] = [i[0]]

    # 构建字典库,User和Item的字典库是一样的
    # rawReviews = bulid_vocbulary(review_dict)
    vocab, user_review2doc, item_review2doc, user_reviews_dict, item_reviews_dict = build_doc(user_reviews_dict, item_reviews_dict)
    word_index = {}
    word_index['<unk>'] = 0
    for i, w in enumerate(vocab.keys(), 1):
        word_index[w] = i
    print("字典库大小{}".format(len(word_index)))

    # user_raw = bulid_vocbulary(user_reviews_dict)
    # item_raw = bulid_vocbulary(item_reviews_dict)

    # # process with tf-idf and stop words
    # user_review2doc = build_doc(user_raw)
    # item_review2doc = build_doc(item_raw)

    print(f"Average user document length: {sum([len(i) for i in user_review2doc])/len(user_review2doc)}")
    print(f"Average item document length: {sum([len(i) for i in item_review2doc])/len(item_review2doc)}")

    print(now())
    u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen, u_pSentLen = countNum(user_reviews_dict)
    print("用户最少有{}个评论,最多有{}个评论，平均有{}个评论, " \
         "句子最大长度{},句子的最短长度{}，" \
         "设定用户评论个数为{}： 设定句子最大长度为{}".format(u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen, u_pSentLen))
    # 商品文本数统计
    i_minNum, i_maxNum, i_averageNum, i_maxSent, i_minSent, i_pReviewLen, i_pSentLen = countNum(item_reviews_dict)
    print("商品最少有{}个评论,最多有{}个评论，平均有{}个评论," \
         "句子最大长度{},句子的最短长度{}," \
         ",设定商品评论数目{}, 设定句子最大长度为{}".format(i_minNum, i_maxNum, i_averageNum, u_maxSent, i_minSent, i_pReviewLen, i_pSentLen))
    print("最终设定句子最大长度为(取最大值)：{}".format(max(u_pSentLen, i_pSentLen)))
    # ########################################################################################################
    maxSentLen = max(u_pSentLen, i_pSentLen)
    minSentlen = 1

    userReview2Index = []
    userDoc2Index = []
    user_iid_list = []

    def padding_text(textList, num):
        new_textList = []
        if len(textList) >= num:
            new_textList = textList[:num]
        else:
            padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
            new_textList = textList + padding
        return new_textList

    def padding_ids(iids, num, pad_id):
        if len(iids) >= num:
            new_iids = iids[:num]
        else:
            new_iids = iids + [pad_id] * (num - len(iids))
        return new_iids

    def padding_doc(doc):
        '''
        doc
        '''
        # DOC_LEN = [len(i) for i in doc]
        # x = np.sort(DOC_LEN)
        # 统计有多少个评论
        # xLen = len(x)
        # 以p覆盖率确定句子长度
        # pDocLen = x[int(P_REVIEW * xLen) - 1]
        pDocLen = DOC_LEN
        new_doc = []
        for d in doc:
            if len(d) < pDocLen:
                d = d + [0] * (pDocLen - len(d))
            else:
                d = d[:pDocLen]
            new_doc.append(d)

        return new_doc, pDocLen

    for i in range(userNum):
        count_user = 0
        dataList = []
        a_count = 0

        textList = user_reviews_dict[i]
        u_iids = user_iid_dict[i]

        u_reviewList = []  # 待添加
        u_reviewLen = []   # 待添加

        user_iid_list.append(padding_ids(u_iids, u_pReviewLen, itemNum+1))

        doc2index = [word_index[w] for w in user_review2doc[i]]

        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) >= minSentlen:
                k = 0
                if len(wordTokens) > maxSentLen:
                    u_reviewLen.append(maxSentLen)
                else:
                    u_reviewLen.append(len(wordTokens))
                for _, word in enumerate(wordTokens):
                    if k < maxSentLen:
                        text2index.append(word_index[word])
                        k = k + 1
                    else:
                        break
            else:
                count_user += 1
                u_reviewLen.append(1)
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            u_reviewList.append(text2index)

        if count_user >= 1:
            print("第{}个用户共有{}个商品评论，经处理后有{}个为空".format(i, len(textList), count_user))

        userReview2Index.append(padding_text(u_reviewList, u_pReviewLen))
        userDoc2Index.append(doc2index)

    # userReview2Index = []
    userDoc2Index, userDocLen = padding_doc(userDoc2Index)
    print(f"user document length: {userDocLen}")

    itemReview2Index = []
    itemDoc2Index = []
    item_uid_list = []
    for i in range(itemNum):
        count_item = 0
        dataList = []
        textList = item_reviews_dict[i]
        i_uids = item_uid_dict[i]
        i_reviewList = []  # 待添加
        i_reviewLen = []  # 待添加
        item_uid_list.append(padding_ids(i_uids, i_pReviewLen, userNum+1))

        doc2index = [word_index[w] for w in item_review2doc[i]]

        for text in textList:
            text2index = []
            # wordTokens = text_to_word_sequence(text)
            wordTokens = text.strip().split()
            if len(wordTokens) >= minSentlen:
                k = 0
                if len(wordTokens) > maxSentLen:
                    i_reviewLen.append(maxSentLen)
                else:
                    i_reviewLen.append(len(wordTokens))
                for _, word in enumerate(wordTokens):
                    if k < maxSentLen:
                        text2index.append(word_index[word])
                        k = k + 1
                    else:
                        break
            else:
                count_item += 1
                i_reviewLen.append(1)
            doc2index.extend(text2index)
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            i_reviewList.append(text2index)
        if count_item >= 1:
            print("第{}个商品共有{}个用户评论,经处理后{}个为空".format(i, len(textList), count_item))
        itemReview2Index.append(padding_text(i_reviewList, i_pReviewLen))
        itemDoc2Index.append(doc2index)

    itemDoc2Index, itemDocLen = padding_doc(itemDoc2Index)
    print(f"item document length: {itemDocLen}")

    print("-"*30)
    print(f"{now()} start writing npy...")
    np.save(f"{save_folder}/train/userReview2Index.npy", userReview2Index)
    np.save(f"{save_folder}/train/user_item2id.npy", user_iid_list)
    np.save(f"{save_folder}/train/userDoc2Index.npy", userDoc2Index)

    np.save(f"{save_folder}/train/itemReview2Index.npy", itemReview2Index)
    np.save(f"{save_folder}/train/item_user2id.npy", item_uid_list)
    np.save(f"{save_folder}/train/itemDoc2Index.npy", itemDoc2Index)
    print(f"{now()} write finised")
    ########################################################################################################

    # #####################################################3,产生w2v############################################
    vocab_item = sorted(word_index.items(), key=itemgetter(1))
    w2v = []
    out = 0
    if PRE_W2V_BIN_PATH:
        pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH, binary=True)
    else:
        pre_word2v = {}
    print(f"{now()} 开始提取embedding")
    for word, key in vocab_item:
        if word in pre_word2v:
            w2v.append(pre_word2v[word])
        else:
            out += 1
            w2v.append(np.random.uniform(-1.0, 1.0, (300,)))
    print("############################")
    print(f"丢失单词数{out}")
    # print w2v[1000]
    print(f"w2v大小{len(w2v)}")
    print("############################")
    w2vArray = np.array(w2v)
    print(w2vArray.shape)
    np.save(f"{save_folder}/train/w2v.npy", w2v)
