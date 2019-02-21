# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 06:05:11 2018

@author: Bohyun
"""

from gensim.test.utils import common_texts
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath

#========================가중치 배열 계산============================
from sklearn.manifold import TSNE
import gensim.models as g
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import squareform, pdist
#=============================표준화==================================
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
  

from konlpy.tag import Twitter
import time
import re

pos_tagger = Twitter()

emoji_pattern = re.compile("["
u"\U0001F600-\U0001F64F"  # emoticons
u"\U0001F300-\U0001F5FF"  # symbols & pictographs
u"\U0001F680-\U0001F6FF"  # transport & map symbols
u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                   "]+", flags=re.UNICODE)

def tokenize(doc):

    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def read_raw_data(filename):

    with open(filename, 'r', encoding='utf-8') as f:

        print('loading data')

        data = []


        for line in f.read().splitlines():
            data_temp = emoji_pattern.sub(r'', line)
         #   print(data_temp)
            data.append(data_temp.split('\t'))
     
        print('pos tagging to token')

        #data = [(tokenize(row[1]), int(row[2])) for row in data[0:]]
        data = [(tokenize(row[1]), int(row[2]),int(row[3]),str(row[4]),str(row[5])) for row in data[0:]]
    return data

def read_categ_data(filename):

    with open(filename, 'r', encoding='utf-8') as f:

        print('loading categ data')


        categ_data = []

        for line in f.read().splitlines():
            data_temp = emoji_pattern.sub(r'', line)
         #   print(data_temp)
            categ_data.append(data_temp.split('\t'))
    #    data = [line.split('\t') for line in f.read().splitlines()]

     
              
        categ_data =[(row[1]) for row in categ_data[0:]]


    return categ_data

'''
def etc_data():
    ##tdm_sum 배열에서 각 항목끼리 빼봄
    ## 차가 min과 max
'''
def build_vocab(tokens):

    print('building vocabulary')

    vocab = dict()
    vocab['#UNKNOWN'] = 0

    vocab['#PAD'] = 1
          
    '''
    
    for t in tokens:
    
        if t not in vocab:
    
            vocab[t] = len(vocab)
    '''
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)

    return vocab



def get_token_id(token, vocab):

    if token in vocab:

        return vocab[token]

    else:

        0 # unkown



def build_input(data, tokens):



    def get_onehot(index, size):

        onehot = [0] * size

        onehot[index] = 1

        return onehot



    print('building input')

    result = []

    for d in data:

        sequence = [get_token_id(t, tokens) for t in d[0]]

        while len(sequence) > 0:

            seq_seg = sequence[:60]

            sequence = sequence[60:]



            padding = [1] *(60 - len(seq_seg))

            seq_seg = seq_seg + padding



            result.append((seq_seg, get_onehot(d[1], 5)))



    return result 







def save_vocab(filename, tokens):

    with open(filename, 'w', encoding='utf-8') as f:

        for v in tokens:
            
            f.write('%s, ' % v)
         

if __name__ == '__main__':

    data = read_raw_data('mm.txt')

    ## data각 행에 [0] 만 사용
    ## data 각 행 length 60 이상이면 배치
    tokens = [t for d in data for t in d[0]]
    print("토큰 전처리")
    toke2=[]
    toke1=[]
    data_length=len(data)
    tokenized=[]
    d_l=-1
    for sentence in data:
        d_l+=1
        line=[]
        tempS=str(sentence[0])
        toke3=tempS.strip().split(', ')
        toke_len=len(toke3)
        for t in range(toke_len):
            toke=toke3[t].replace("'","").replace("[","").replace("]","")
            if "Noun" in toke:
                line.append(toke)
            elif "Verb" in toke:
                line.append(toke)
            elif "Adjective" in toke:
                line.append(toke)                      
        tokenized.append(list(line))

    r = [tokenized]
    print("token word2vec")
#===============Word2Vec=========================
    model=Word2Vec(tokenized, size=100, window = 2, min_count=20, workers=4, iter=100, sg=1)

    model_name = 'word2vec_model'

    model.save(model_name)

    print("modeling word2vec")
    word_vocab = model.wv.vocab
#========================가중치 배열 계산============================
  
    model_name = 'word2vec_model'
    model = g.Doc2Vec.load(model_name)
    
    vocab = list(model.wv.vocab)
    X = model[vocab]
    
    df = pd.DataFrame(model[vocab], index = vocab)

   #퀘스트 난이도 이벤트 미션 퀘
   #캐릭터 스킬   소환 유
   #연출 스토리 그래픽 디자인 이펙트
   # 계정 아이디 연동 구글 로그인
    category_in= ["결제/Noun","구입/Noun","구매/Noun","현질/Noun","환불/Noun", #결제
                  '계정/Noun','아이디/Noun','연동/Noun','구글/Noun','로그인/Noun',#계정
               "버그/Noun","서버/Noun","접속/Noun","로딩/Noun","렉/Noun", #서버
               "화려하다/Adjective","그래픽/Noun","퀄리티/Noun","사운드/Noun","디자인/Noun", #연출, 액션
               "스킬/Noun","너프/Noun","영웅/Noun","캐릭터/Noun","신캐/Noun", #캐릭터 체력
              
               "용량/Noun","다운/Noun","앱/Noun","실행/Noun","설치/Noun", #시스템
               "광고/Noun","신고/Noun","채팅/Noun","욕/Noun","처벌/Noun"]#기타, 불만
               
    ##"맵/Noun","이벤트/Noun","퀘스트/Noun","스테이지/Noun","미션/Noun", #구성 , 기획
    ## "아이템/Noun","파밍/Noun","장비/Noun","상점/Noun","인벤/Noun", #아이템, 무기
    category = ['결제', '계정', '서버','연출','캐릭터','시스템','기타']

    
    cate_length=len(category_in) #카테고리 수 수정한 후 갯수20개
    
    df_eucli = pd.DataFrame(squareform(pdist(df.iloc[:, 1:])),columns=df.index.unique(), index=df.index.unique())
#======================정규분포===================================
    df_theme=df_eucli[category_in]
    
    f0_list_name = ["결제/Noun","구입/Noun","구매/Noun","현질/Noun","환불/Noun"]
    f1_list_name = ['계정/Noun','아이디/Noun','연동/Noun','구글/Noun','로그인/Noun']
    f2_list_name = ["버그/Noun","서버/Noun","접속/Noun","로딩/Noun","렉/Noun"]

    f3_list_name = ["배경/Noun","그래픽/Noun","퀄리티/Noun","사운드/Noun","디자인/Noun"]
    f4_list_name = ["스킬/Noun","너프/Noun","영웅/Noun","캐릭터/Noun","신캐/Noun"] 
   # f6_list_name = ["아이템/Noun","파밍/Noun","장비/Noun","상점/Noun","인벤/Noun"]
    f5_list_name = ["용량/Noun","다운/Noun","앱/Noun","실행/Noun","설치/Noun"]
    f6_list_name = ["광고/Noun","신고/Noun","채팅/Noun","욕/Noun","처벌/Noun"]
    
    # 새로 카테고리 배열들 합 구하기스
    df_f1 = df_theme
    df_temp=df_theme

    
  
    import numpy as np
    print("가중치 행렬 계산")
    #df_pow=pd.DataFrame.pow(df_theme,fill_value=None)
    df_pow=np.power(df_theme, 2)
    df_pow.loc['temp'] = [0 for n in range(cate_length)]#temp 행 추가
    df_exp=np.exp(-df_pow/100)
    df_exp1 =  df_exp.drop("temp")
    df_pow =  df_pow.drop("temp", axis = 0)

    df_length=len(df_exp1.index)

    sum=0
    for j in range(cate_length):         
        for i in range(df_length):
            sum+=df_exp1.iloc[i,j]
        df_mean=sum/df_length
        tuneweight=0.5-df_mean
        for k in range(df_length):
            df_exp1.iloc[i,j]-=tuneweight
        sum=0
  #  df_exp1 이 가중치행렬

#=====================TDM===================================
    catego_data = read_categ_data('mm.txt')
    catego_data = pd.DataFrame(catego_data) 
    catego_data.columns=["content"] 
    d_dataFrame=pd.DataFrame(catego_data)   
    df_senFrame=d_dataFrame[["content"]]
    categ_data = pd.DataFrame(catego_data)

    d_length=len(categ_data)
    #len(data) #총 갯수 25535개
    print("TDM 배열")
    TDM=[[0 for col in range(60)] for row in range(d_length)]
   # print(len(TDM))
    w=[0]*60
    #f1-f6 col, 각 리뷰 row
    tdm_sum=[[0 for col in range(cate_length+1)] for row in range(d_length)]
    for i in range(d_length):
        #print(df_senFrame.iloc[[i]])
        #data를 쓰자

        df_string=str(data[i][0])
        w=df_string.split(', ')
        ##df마지막에 length추가하기 TODO
        
        token_length=len(w)
        if(token_length>=60):
            token_length=60
        df_flag=[0]*(cate_length)
        print("TDM 단어 존재 확인")
        if(token_length>5):
            for j in range(token_length):
         
                tempWord=w[j].replace("'","").replace("[","").replace("]","")
                ##단어 존재하는지 확인
                for k in range(cate_length):
                    try:
                        df_value=df_exp1.loc[tempWord,category_in[k]]
                        #1로배정
                    except KeyError as e:
                        TDM[i][j]=0
                    else:
                        if(df_flag[k]==1):
                            df_value=0
                            continue
                        else:
                            df_flag[k]=1
                            TDM[i][j]=1 ##몇번 나왔는지까지 
                            tdm_sum[i][k]+=(df_value) 
                            df_value=0
                df_flag=[0 for m in range(cate_length)]
        else:##TOO SHORT
            tdm_sum[i][cate_length]=1000
                
    print("TDM_SUM 계산")           
    TDM_SUM=[[0 for col in range(9)] for row in range(d_length)]
    STD_SCALE=[[0 for col in range(9)] for row in range(d_length)]
    scaler = StandardScaler()
    temp_sum=[0]*9
    for line in range(d_length):
        std_scaler = [0]*9

        for col in range(cate_length):
            if ((col>=0 and col<=4)): #결제
                temp_sum[0]+=tdm_sum[line][col]
            elif ((col>=5 and col<=9)): #계정
                temp_sum[1]+=tdm_sum[line][col]
            elif ((col>=10 and col<=14)): #서버
                temp_sum[2]+=tdm_sum[line][col]
            elif ((col>=15 and col<=19)): #연출
                temp_sum[3]+=tdm_sum[line][col]
            elif ((col>=20 and col<=24)): #캐릭터
                temp_sum[4]+=tdm_sum[line][col]
            elif ((col>=25 and col<=29)): #시스템
                temp_sum[5]+=tdm_sum[line][col]
            elif ((col>=30 and col<=34)): #기타
                temp_sum[6]+=tdm_sum[line][col]
            elif ((col>=35 and col<=39)): #기타
                temp_sum[7]+=tdm_sum[line][col]
         ##   elif ((col==40)): #TOO_SHorT
          ##      temp_sum[8]+=tdm_sum[line][col] 
          
        TDM_SUM[line][0]=temp_sum[0]
        TDM_SUM[line][1]=temp_sum[1]
        TDM_SUM[line][2]=temp_sum[2]
        TDM_SUM[line][3]=temp_sum[3]
        TDM_SUM[line][4]=temp_sum[4]
        TDM_SUM[line][5]=temp_sum[5]
        TDM_SUM[line][6]=temp_sum[6]
        TDM_SUM[line][7]=temp_sum[7]
       # TDM_SUM[line][8]=temp_sum[8]
       # TDM_SUM[line][9]=temp_sum[9]
        temp_sum=[0]*9  
#        STD_SCALE[line]=ss.zscore(TDM_SUM[line])
#=======================표준화===================================     




#=======================결과===================================    
    print("카테고리 max")
    categ_max =[]
    categ_arr=[]
    categ_arr = categ_data.as_matrix()
    f=[k for k in range(9)]
    for col in range(9):
        f[col]=[]
    cate_result=[]
    for i in range(d_length):
        if (max(TDM_SUM[i]) != 0): #짧지않다
            import copy 
            desc = copy.deepcopy(TDM_SUM)
            desc[i].sort(reverse=True)#내림차순
            #임계값
            threshold=0.35
            max_score=desc[i][0]
            cate_result=[]
            max_index=TDM_SUM[i].index(desc[i][0])
            if max_index==0 :
                categ_result = "결제"
                f[max_index].append([categ_arr[i]])
            elif max_index == 1:
        
                categ_result = "계정"
                f[max_index].append([categ_arr[i]])
            elif max_index == 2:
          
                categ_result = "서버"
                f[max_index].append([categ_arr[i]])
            elif max_index == 3:
        
                categ_result = "연출"
                f[max_index].append([categ_arr[i]])
            elif max_index == 4:
   
                categ_result = "캐릭터"
                f[max_index].append([categ_arr[i]])
            elif max_index == 5:
       
                categ_result = "시스템"
                f[max_index].append([categ_arr[i]])
            elif max_index == 6:
          
                categ_result = "기타"
                f[max_index].append([categ_arr[i]])
            elif max_index== 7:
          
                categ_result = "TOO_SHORT"
                f[max_index].append([categ_arr[i]])
            cate_result.append(categ_result)
            
            for col in range(1,2):   
                ##0이 중복
                if((max_score-desc[i][col]) <= threshold): 
                    max_index=TDM_SUM[i].index(desc[i][col])
                    if max_index==0 :
                        categ_result = "결제"
                        f[max_index].append([categ_arr[i]])
                    elif max_index == 1:
                        categ_result = "계정"
                        f[max_index].append([categ_arr[i]])
                    elif max_index == 2:
                        categ_result = "서버"
                        f[max_index].append([categ_arr[i]])
                    elif max_index == 3:
                        categ_result = "연출"
                        f[max_index].append([categ_arr[i]])
                    elif max_index == 4:
                        categ_result = "캐릭터"
                        f[max_index].append([categ_arr[i]])
                    elif max_index == 5:
                        categ_result = "시스템"
                        f[max_index].append([categ_arr[i]])
                    elif max_index == 6:
                  
                        categ_result = "기타"
                        f[max_index].append([categ_arr[i]])
                    elif max_index== 7:
                  
                        categ_result = "TOO_SHORT"
                        f[max_index].append([categ_arr[i]])
                    cate_result.append(categ_result)
            print(i,': ',categ_arr[i],' ',str(cate_result))
            

    categ_data_f0 = pd.DataFrame(f[0])
    categ_data_f1 = pd.DataFrame(f[1])
    categ_data_f2 = pd.DataFrame(f[2])
    categ_data_f3 = pd.DataFrame(f[3])
    categ_data_f4 = pd.DataFrame(f[4])
    categ_data_f5 = pd.DataFrame(f[5])
    categ_data_f6 = pd.DataFrame(f[6])
    categ_data_f7 = pd.DataFrame(f[7])
 #   categ_data_f8 = pd.DataFrame(f[8])
    #categ_data_f9 = pd.DataFrame(f[9])  
    '''
    categ_data_f0.columns=["결제"]
    categ_data_f1.columns=["계정"]
    categ_data_f2.columns=["서버"]
    categ_data_f3.columns=["구성"]
    categ_data_f4.columns=["연출"]
    categ_data_f5.columns=["캐릭터"]
    categ_data_f6.columns=["아이템"]
    categ_data_f7.columns=["시스템"]
    categ_data_f8.columns=["기타"]
    
    categ_data_f9.columns=["TOO_SHORT"]
    '''
  ##'결제', '계정', '서버','구성','연출','캐릭터','아이템','시스템','기타'
    print("******** 결제 positive **************")
    print(categ_data_f0.head(10))
    print("******** 계정 positive **************")
    print(categ_data_f1.head(10))
    print("******** 서버 positive **************")
    print(categ_data_f2.head(10))
   # print("******** 구성 positive **************")
   # print(categ_data_f3.head(10))
    
    print("******** 연출 positive **************")

    print(categ_data_f3.head(10))
    print("******** 캐릭터 positive **************")
    print(categ_data_f4.head(10))
    print("******** 시스템 positive **************")
    print(categ_data_f5.head(10))
    print("******** 기타 positive **************")
    print(categ_data_f6.head(10))
    print("******** TOO_SHORT positive **************")
    print(categ_data_f7.head(10))
   

    
    print("data 총 수: ", len(tdm_sum) )
    print( "category0 :", len(categ_data_f0))
    print( "category1 :", len(categ_data_f1))
    print( "category2 :", len(categ_data_f2))
    print( "category3 :", len(categ_data_f3))   
    print( "category4 :", len(categ_data_f4))
    print( "category5 :", len(categ_data_f5))
    print( "category6 :", len(categ_data_f6))
    print( "category7 :", len(categ_data_f7))
 #   print( "category8 :", len(categ_data_f8))
    #print( "category9 :", len(categ_data_f9))
                    

        
    
    ## 단어없을 때 벡터 중 가장 높은 값으로 대체 flag
  
