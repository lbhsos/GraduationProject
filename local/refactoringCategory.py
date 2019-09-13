# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:50:14 2019

@author: Bohyun
"""

#========================Word2Vec===================================
from gensim.models import Word2Vec 
#========================가중치 배열 계산============================
import gensim.models as g
import pandas as pd
from scipy.spatial.distance import squareform, pdist
#=============================표준화=================================
from konlpy.tag import Twitter
import re
import numpy as np
import copy 

category = ['결제', '계정', '서버','구성','연출','캐릭터','시스템','기타']
pay_words = ["결제/Noun","구입/Noun","구매/Noun","현질/Noun","환불/Noun"] #결제 0
id_words = ['계정/Noun','아이디/Noun','연동/Noun','구글/Noun','로그인/Noun'] #계정 1 
server_words = ["버그/Noun","서버/Noun","접속/Noun","로딩/Noun","렉/Noun"] #서버 2
config_words = ["맵/Noun","이벤트/Noun","퀘스트/Noun","스테이지/Noun","미션/Noun"] #구성, 기획 3 
production_words = ["배경/Noun","그래픽/Noun","퀄리티/Noun","사운드/Noun","디자인/Noun"] #연출, 액션 4 
character_words = ["스킬/Noun","너프/Noun","영웅/Noun","캐릭터/Noun","캐릭/Noun"] #캐릭터, 체력 5
sys_words = ["용량/Noun","다운/Noun","앱/Noun","실행/Noun","설치/Noun"] #시스템 6
dissatis_words = ["광고/Noun","신고/Noun","채팅/Noun","욕/Noun","처벌/Noun"] #불만 7

categorySubject= pay_words + id_words + server_words +config_words + production_words + character_words + sys_words + dissatis_words
allCategoryWordsLength=len(categorySubject) 
tempReviewLength=300 #카테고리 분류 할 리뷰 개수


def tokenize(doc):
    pos_tagger = Twitter()
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def removeEmoji():
    emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                   "]+", flags=re.UNICODE)
    return emoji_pattern

def readRawData(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        print('loading data')
        dataSplitedByTab = []
        tokenizedReviews = []
        reviewSentencesWithInfo = []
        for line in f.read().splitlines():
            data_temp = removeEmoji().sub(r'', line)
            dataSplitedByTab.append(data_temp.split('\t'))   
        print('pos tagging to token')
#        tokenizedReviews = [(tokenize(row[1])) for row in dataSplitedByTab[0:]]
#        reviewSentencesWithInfo =[(row[1]) for row in dataSplitedByTab[0:]]
        tokenizedReviews = []
        reviewSentencesWithInfo = []
        for row in dataSplitedByTab[0:] :
            row_1 = row[1]
            tokenizedReviews.append(tokenize(row_1))
            reviewSentencesWithInfo.append(row_1)
            
    return (tokenizedReviews, reviewSentencesWithInfo)

def removeNounVerbAdj(tokenizedReviews):
#    tokens = [t for d in tokenizedReview for t in d]
   # data_length=len(tokenizedReview)
    preprocessedTokens=[]
    
    for sentence in tokenizedReviews:
        line=[]
        tempSentence=str(sentence)
        sentenceWithoutComma=tempSentence.strip().split(', ')
        tokenLength=len(sentenceWithoutComma)
        for t in range(tokenLength):
            token=sentenceWithoutComma[t].replace("'","").replace("[","").replace("]","")
            if "Noun" in token:
                line.append(token)
            elif "Verb" in token:
                line.append(token)
            elif "Adjective" in token:
                line.append(token)                      
        preprocessedTokens.append(list(line))
    return preprocessedTokens

def createWord2VecModel(tokenized): #word2vec 모델 생성 및 vocab 생성
    model=Word2Vec(tokenized, size=100, window = 2, min_count=20, workers=4, iter=100, sg=1)
    model_name = 'word2vec_model'
    print("modeling word2vec")
    model.save(model_name)
    return model_name

def createWordVectorMatrix(model_name): #word2vec의 결과 단어벡터 행렬
    print("build vocab")
    model = g.Doc2Vec.load(model_name) 
    vocab = list(model.wv.vocab)
    wordVectorMatirx = pd.DataFrame(model[vocab], index = vocab)
    return wordVectorMatirx

def createEuclidianDistanceMatrix(wordVectorMatirx) :
    eucliDistancMatrix = pd.DataFrame(squareform(pdist(wordVectorMatirx.iloc[:, 1:])),columns=wordVectorMatirx.index.unique(), index=wordVectorMatirx.index.unique())
    return eucliDistancMatrix

def createWeightingMatrix(eucliDistancMatrix): #정규 분포 사용하여 가중치 행렬로 변환
    matrixOnlyWithSubject=eucliDistancMatrix[categorySubject]
    # 새로 카테고리 배열들 합 구하기스
    print("가중치 행렬 계산")
    matrixSquare=np.power(matrixOnlyWithSubject, 2)
    #temp행 추가 
    matrixSquare.loc['temp'] = [0 for n in range(allCategoryWordsLength)]
    exponentialMatrix=np.exp(-matrixSquare/100)
    weightingMatrix =  exponentialMatrix.drop("temp")
    matrixSquare =  matrixSquare.drop("temp", axis = 0)
    matrixLength=len(weightingMatrix.index)
    sum=0
    for j in range(allCategoryWordsLength):         
        for i in range(matrixLength):
            sum+=weightingMatrix.iloc[i,j]
        matrixMeanValue=sum/matrixLength
        tuneweight=0.5-matrixMeanValue
        for k in range(matrixLength):
            weightingMatrix.iloc[i,j]-=tuneweight
        sum=0

    return weightingMatrix

def createTermDocumentMatrix(reviewSentencesWithInfo):
    print("TDM 배열")
    reviewSentencesWithInfo = pd.DataFrame(reviewSentencesWithInfo) 
    reviewSentencesWithInfo.columns=["content"] 
    reviewSentences = pd.DataFrame(reviewSentencesWithInfo)
    return reviewSentences

##내적 구하는 행렬
def computeInnerProduct(tokenizedReviews, weightingMatrix):
    TDM=[[0 for col in range(60)] for row in range(tempReviewLength)] 
    wordToken=[0]*60
    minimumTokenLength=5
    #f1-f6 col, 각 리뷰 row
    sumOfCateBranchArr=[[0 for col in range(allCategoryWordsLength+1)] for row in range(tempReviewLength)]
    for i in range(tempReviewLength):
        tokenizedReview=str(tokenizedReviews[i])
        wordToken=tokenizedReview.split(', ')
        ##df마지막에 length추가하기 TODO      
        tokenLength=len(wordToken)
        if(tokenLength>=60):
            tokenLength=60
        flagForAdvent=[0]*(allCategoryWordsLength)
        if(tokenLength>minimumTokenLength):
            for j in range(tokenLength):
                tempWord=wordToken[j].replace("'","").replace("[","").replace("]","")
                ##단어 존재하는지 확인
                for eachCategoryWord in range(allCategoryWordsLength):
                    try:
                        value=weightingMatrix.loc[tempWord,categorySubject[eachCategoryWord]]
                        #1로배정
                    except KeyError as e:
                        TDM[i][j]=0
                    else:
                        if(flagForAdvent[eachCategoryWord]==1): # 
                            value=0
                            continue
                        else:
                            flagForAdvent[eachCategoryWord]=1
                            TDM[i][j]=1 ##only count one
                            sumOfCateBranchArr[i][eachCategoryWord]+=(value) 
                            value=0
                flagForAdvent=[0 for m in range(allCategoryWordsLength)] ##initialize
        else:##TOO SHORT
            sumOfCateBranchArr[i][allCategoryWordsLength]=1000
    return sumOfCateBranchArr



def getSentenceCategorySum(sumOfCateBranchArr):
    sumOfCategory=[[0 for col in range(9)] for row in range(tempReviewLength)]
    for line in range(tempReviewLength):
        for i in range(0,9) :
            sumOfCategory[line][i] += sum(sumOfCateBranchArr[line][i*5: min((i+1)*5,41)])
    return sumOfCategory

def classificate(sumOfCategory): 
    print("카테고리 max으로 분류하기")
    reviewSentencesAsArr=[]
    reviewSentencesAsArr = reviewSentences.as_matrix()
    categorizedReviews=[k for k in range(9)]
    for col in range(9):
        categorizedReviews[col]=[]
    
    for i in range(tempReviewLength):
        if (max(sumOfCategory[i]) != 0):
            desc = copy.deepcopy(sumOfCategory)
            desc[i].sort(reverse=True)
            threshold=0.35
            max_score=desc[i][0]
            finalCategory=[]
            max_index=sumOfCategory[i].index(desc[i][0])
            categoryResult=findCategoryByMaxIndex(categorizedReviews,max_index, reviewSentencesAsArr,i)
            finalCategory.append(categoryResult)
            for col in range(1,2):   
                if((max_score-desc[i][col]) <= threshold): 
                    max_index=sumOfCategory[i].index(desc[i][col])
                    finalCategory.append(categoryResult)
                    categoryResult=findCategoryByMaxIndex(categorizedReviews,max_index, reviewSentencesAsArr,i)
                    finalCategory.append(categoryResult)
    return categorizedReviews

def findCategoryByMaxIndex(categorizedReviews,max_index,reviewSentencesAsArr,i):
    if max_index==0 :
        categoryResult = "결제"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index == 1:
        categoryResult = "계정"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index == 2:
        categoryResult = "서버"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index == 3:
        categoryResult = "구성"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index == 4:
        categoryResult = "연출"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index == 5:
        categoryResult = "캐릭터"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index == 6:
        categoryResult = "시스템"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index== 7:
        categoryResult = "기타"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    elif max_index == 8:
        categoryResult = "TOO_SHORT"
        categorizedReviews[max_index].append([reviewSentencesAsArr[i]])
    return categoryResult

def result(f):
    result_review = []
    for i in range(9):
        result_review.append(pd.DataFrame(f[i]))
    return result_review   

def print_result(result_review):
    for i in range(9):
        print("리뷰 출력", result_review[i].head(10))
        print("len", len(result_review[i]))

if __name__ == '__main__':
    tokenizedReviews, reviewSentencesWithInfo = readRawData('./mm.txt') #파일 읽어서 데이터 로딩
    preprocessedTokens = removeNounVerbAdj(tokenizedReviews) #토큰 전처리
    word2vecModel = createWord2VecModel(preprocessedTokens) #워드투벡 모델 생성
    wordVectorMatirx = createWordVectorMatrix(word2vecModel) #워드투벡 결과로 나온 단어벡터 행렬                  
    eucliDistancMatrix = createEuclidianDistanceMatrix(wordVectorMatirx) #유클리디안 거리 이용하여 거리행렬로 변환
    weightingMatrix = createWeightingMatrix(eucliDistancMatrix) #정규 분포 사용해서 가중치행렬로 변환
    reviewSentences = createTermDocumentMatrix(reviewSentencesWithInfo) #TDM 구축
    sumOfCateBranchArr = computeInnerProduct(tokenizedReviews, weightingMatrix) #내적
    sumOfCategory = getSentenceCategorySum(sumOfCateBranchArr) #TDM 합 구하기
    categorizedReviews=classificate(sumOfCategory) #카테고리 분류
    result_review = result(categorizedReviews)
    print_result(result_review)
    
    '''
    df_exp1 = weighted_matrix(df_theme) #정규 분포 사용해서 가중치행렬로 변환
    categ_data = term_document_matrix(content) #TDM 구축
    tdm_sum = compute_inner_product(data, df_exp1) #내적
    TDM_SUM = compute_TDM_sum(tdm_sum) #TDM 합 구하기
    f,categ_arr, cate_result=classification(TDM_SUM) #카테고리 분류
    result_review = result(f)
    print_result(result_review)
    '''
    
