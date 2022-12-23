import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import  natsorted
import pandas as pd
import math
import numpy as np
import re as ree
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

print('-----------------------------------------------------Toknization and Stopwords part-----------------------------------------------')
stop_words = stopwords.words('english')
files_name = natsorted(os.listdir('collection'))

document_of_terms = []
for files in files_name:
    with open(f'collection/{files}' , 'r') as f:
         document = f.read()
    tokenized_documents = word_tokenize(document)
    terms = [ ]
    for word in tokenized_documents:
         if word not in stop_words:
             terms.append(word)
    document_of_terms.append(terms)

for l in (document_of_terms):
    print(l,'\n')
print()
print()
print("-----------------------------------------------------positional index  part-----------------------------------------------")
print()
stop_words = stopwords.words('english')
stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')

files_name = natsorted(os.listdir('collection'))

document_of_terms = []
for files in files_name:
    with open(f'collection/{files}' , 'r') as f:
         document = f.read()
    tokenized_documents = word_tokenize(document)
    terms = [ ]
    for word in tokenized_documents:
         if word not in stop_words:
             terms.append(word)
    document_of_terms.append(terms)


document_number = 1
positional_index = { }
for document in document_of_terms:
        for positional,term in enumerate(document):

                 if term in positional_index:

                       positional_index[term][0] =positional_index[term][0] +1

                       if document_number in positional_index[term][1]:
                           positional_index[term][1][document_number].append(postional)
                       else:
                             positional_index[term][1][document_number] = [positional]

                 else:
                       positional_index[term] = []
                       positional_index[term].append(1)
                       positional_index[term].append({})
                       positional_index[term][1][document_number] = [positional]


        document_number +=1
for l, k in sorted(positional_index.items()):
    print(l, ":", k,'\n')

print()
print()
# print("-----------------------------------------------------Query search part-----------------------------------------------")
# print()
#
#
# while(1):
#     query = input("Enter the phrase you want to search for:")
#
#     final_list = [ [] for i in range (10) ]
#
#     for word in query.split():
#             for key in positional_index[word][1].keys():
#
#                     if final_list[key-1] !=[ ]:
#                          if final_list[key-1][-1] ==positional_index[word][1][key][0]-1:
#                                 final_list[key-1].append(positional_index[word][1][key][0])
#                     else:
#                          final_list[key-1].append(positional_index[word][1][key][0])
#
#     print("The matched documents for the phrase query:")
#     for position , list in enumerate(final_list,start=1):
#
#                       if len(list)==len(query.split()):
#                             print(position)


print("--------------------------------------------------------------#Ranking part------------------------------------------------------------")
print()
all_words = []
for doc in document_of_terms:
    for word in doc:
        all_words.append(word)

def get_tf(doc):
  words_found=dict.fromkeys(all_words,0)
  for word in doc:
    words_found[word]+=1
  return words_found
term_freq = pd.DataFrame(get_tf(document_of_terms[0]).values(),index=get_tf(document_of_terms[0]).keys())

for i in range(1,len(document_of_terms)):
    term_freq[i] =get_tf(document_of_terms[i]).values()

term_freq.columns = ['doc'+str(i)for i in range(1, 11)]

print('\n---------------------------------------(1) Term frequency for each word in each document------------------------------------------------\n')
print(term_freq)
print('\n----------------------------------------------------------------------------------------------------------------------------------------\n')

#getting tf weight fo each term                                     1+log(tf)
def get_weighted_term_freq(x):
    if x>0:
        return math.log(x)+1
    return 0
for i in range(1,len(document_of_terms)+1):
    term_freq['doc'+str(i)]= term_freq['doc'+str(i)].apply(get_weighted_term_freq)
print('\n---------------------------------------(2) Term frequency weight for each word in each document----------------------------------------\n')
print(term_freq)
print('\n----------------------------------------------------------------------------------------------------------------------------------------\n')


#getting tf and idf for each term                            idf = log10 N(num of docs) / df
tfd = pd.DataFrame(columns=['df', 'idf'])
for i in range (len(term_freq)):
    frequency = term_freq.iloc[i].values.sum()
    tfd.loc[i, 'df'] = frequency
    tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))
tfd.index = term_freq.index
print('\n------------------------------------------------------(3) DF and IDF for each term-----------------------------------------------------\n')
print(tfd)
print('\n----------------------------------------------------------------------------------------------------------------------------------------\n')

#Tf*idf matrix                                              log(1+tf) * log 10(N/df)
term_freq_inve_doc_freq = term_freq.multiply(tfd['idf'], axis=0)
print('\n---------------------------------------------------------------TF.IDF Matrix------------------------------------------------------------\n')
print(term_freq_inve_doc_freq)
print('\n----------------------------------------------------------------------------------------------------------------------------------------\n')


#getting length                                     sqrt(idf^2 (for each term))
document_length = pd.DataFrame()
def get_docs_length(col):
    return  np.sqrt(term_freq_inve_doc_freq[col].apply(lambda x: x**2).sum())
for column in term_freq_inve_doc_freq.columns:
    document_length.loc['',column+' Length']= get_docs_length(column)
print('\n--------------------------------------------------------The length for each document----------------------------------------------------\n')
print(document_length)
print('\n----------------------------------------------------------------------------------------------------------------------------------------\n')




#getting normalized tf.idf                      tf.idf/doc lenght

normalized_term_freq_idf = pd.DataFrame()
def get_normalized(col,x):
    try:
        return x / document_length[col+' Length'].values[0]
    except:
        return 0

for column in term_freq_inve_doc_freq.columns:
    normalized_term_freq_idf[column]=term_freq_inve_doc_freq[column].apply(lambda x : get_normalized(column,x))
print('\n--------------------------------------------------------The Normalized term frequency------------------------------------------------\n')

print(normalized_term_freq_idf)
print('\n--------------------------------------------------------------------------------------------------------------------------------------\n')



###Cosin Similarity
print("--------------------------------------------------------------#Query search part-------------------------------------------------------")
print()
def que(q):
    list = [[] for i in range(10)]
    for term in q.split():
        if term in positional_index.keys():
            for key in positional_index[term][1].keys():
             if list[key-1]!=[]:
                 if list[key-1][-1] == positional_index[term][1][key][0]-1:
                     list[key-1].append(positional_index[term][1][key][0])
             else:
                 list[key-1].append(positional_index[term][1][key][0])
    positions = []
    for pos ,list in enumerate(list,start=1):
        if(len(list))==len(q.split()):
            positions.append('doc'+str(pos))
    return positions

def get_w_tf(x):
    try:
        return math.log10(x) + 1
    except:
        return 0
while (1):
    query = input("Enter the phrase you want to search for:")
    queryy = query
    docFound=que(queryy)
    if docFound==[]:
        print("Not Found")
        break

    else:
        print('\n')
        final_list = [[] for i in range(10)]

        for word in query.split():
            for key in positional_index[word][1].keys():

                if final_list[key - 1] != []:
                    if final_list[key - 1][-1] == positional_index[word][1][key][0] - 1:
                        final_list[key - 1].append(positional_index[word][1][key][0])
                else:
                    final_list[key - 1].append(positional_index[word][1][key][0])

        qDet = pd.DataFrame(index= normalized_term_freq_idf.index)

        qDet['tf'] = [1 if x in queryy.split() else 0 for x in normalized_term_freq_idf.index]
        qDet['w_tf'] = qDet['tf'].apply(lambda x: get_w_tf(x))
        product = normalized_term_freq_idf.multiply(qDet['w_tf'], axis=0)
        qDet['idf'] = tfd['idf'] * qDet['w_tf']
        qDet['tf_idf'] = qDet['w_tf'] * qDet['idf']
        qDet['norm'] = 0
        for i in range(len(queryy)):
            qDet['norm'].iloc[i] = float(qDet['idf'].iloc[i]) / math.sqrt(sum(qDet['idf'].values ** 2))
        product2 = product.multiply(qDet['norm'], axis=0)
        scores = {}
        for col in docFound:
                scores[col] = product2[col].sum()
        product2=product2[scores.keys()].loc[queryy.split()]
        query_lenght=math.sqrt(sum([x ** 2 for x in qDet['idf'].loc[queryy.split()]]))

        print('\n--------------------------------------------------------Query Details-----------------------------------------------------------------\n')
        print(qDet)
        print("\n--------Query Product------:")
        print(product2)
        print("\n--------Query Product Sum------:")
        print(product2.sum().to_string())
        print("\nQuery Lenght:",query_lenght)
        print('\n--------------------------------------------------------------------------------------------------------------------------------------\n')

        print("The matched documents for the phrase query:")
        for position, list in enumerate(final_list, start=1):

            if len(list) == len(query.split()):
                print("Doc #",position)

        document=[]
        for files in files_name:
            with open(f'collection/{files}' , 'r') as f:
                 document.append (f.read())

        vector = TfidfVectorizer()
        x = vector.fit_transform(document)
        x = x.T.toarray()
        df = pd.DataFrame(x,index=vector.get_feature_names_out())
        q=[query]
        q_vector = (vector.transform(q).toarray().reshape(df.shape[0]))
        similarity = { }
        for i in range (10):
            similarity[i] = np.dot(df.loc[: , i].values, q_vector)/np.linalg.norm(df.loc[: , i])*np.linalg.norm(q_vector)
        similarity_sorted = sorted(similarity.items(),  key=lambda  x: x[ 1 ])
        similarity_sorted_desc = sorted(similarity.items(),  key=lambda  x: x[ 1 ],reverse=True)




        print("\n-----------------------------------------------------Cosin similarity for the phrase query---------------------------------------------\n")
        scores = {}
        for col in product2.columns:
            if 0 in product2[col].loc[queryy.split()].values:
                pass
            else:
                scores[col] = product2[col].sum()
        scores_sorted_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(scores)
        print("\nDocuments ranking based on the cosin similarity:")
        for document,score in scores_sorted_desc:
            print(document)


        c = int(input("Enter #1 to Enter another phrase or #0 to Exit:"))
        if(c==0):break