import os,glob,re
from string import punctuation
from collections import defaultdict
from collections import Counter
import numpy as np
from collections import OrderedDict
import copy
from sklearn import metrics
import operator
from textblob.classifiers import NaiveBayesClassifier

list_negative_word = []
list_positive_word = []
Unique_list_positive_word = []
Unique_list_negative_word = []
cwd = os.getcwd() 
count_positive_ID = 0
count_negative_ID = 0
Negative_Word_Dict = defaultdict()
Positive_Word_Dict = defaultdict()
Sentence_Sentiment = {}
Evaluation_Dict = OrderedDict()

frequent_word_list = ['gamit', 'ako', 'sa', 'ha','lagi', 'ang', 'iyong','ginamit','yun', 'ito', 'bakit', 'ganon','ganoon','ito','mga']
ID_Val = []

for line in open("busNegT-train.txt",encoding="utf8").readlines():
    if line.strip():
        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
        for each_word in word_split:
                        if re.match(r'[IDidIdiD].*-[0-9].*',each_word):
                            count_negative_ID+=1
                            ID_Val.append(each_word)
                            continue
                        elif re.match(r'(\d+)\.(\d+)+',each_word):
                            continue
                        elif re.match(r'(\d+)+',each_word):
                            continue 
                        elif re.match(r'$(\d+)+',each_word):
                            continue  
                        elif re.match(r'[A-Za-z]*\.+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\?+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\!+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'-+',each_word):
                            continue
                        elif re.match(r'[A-Za-z]*\)+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'\(+[A-Za-z]*',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\,+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                            
                        if each_word in frequent_word_list:
                            continue
                        else:
                            list_negative_word.append(each_word)

List_Size_Negative = len(list_negative_word)
Negative_Word_Dict =  Counter(list_negative_word)
Denominator_Sum_Neg = sum(Negative_Word_Dict.values())
Unique_list_negative_word = set(list_negative_word)
Unique_List_Size_Negative = len(Unique_list_negative_word)
list_positive_word = []
for line in open("busPosT-train.txt",encoding="utf8").readlines():
    if line.strip():
        word_split = line.split()
        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()
        for each_word in word_split:
                        if re.match(r'ID-[0-9].*',each_word):
                            count_positive_ID+=1
                            ID_Val = each_word
                            continue
                        elif re.match(r'(\d+)\.(\d+)+',each_word):
                            continue
                        elif re.match(r'(\d+)+',each_word):
                            continue 
                        elif re.match(r'$(\d+)+',each_word):
                            continue  
                        elif re.match(r'[A-Za-z]*\.+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\?+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\!+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'-+',each_word):
                            continue
                        elif re.match(r'[A-Za-z]*\)+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'\(+[A-Za-z]*',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\,+',each_word):
                            each_word = re.sub(r'[^\w\s]','',each_word)
                        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',each_word):
                            list_word = each_word.split('/')
                            if list_word[0] not in frequent_word_list:
                                list_negative_word.append(list_word [0])
                            else:
                                continue
                            
                            if list_word[1] not in frequent_word_list:
                                list_negative_word.append(list_word [1])
                            else:
                                continue
                        if each_word in frequent_word_list:
                            continue
                        else:
                            list_positive_word.append(each_word)
#List of all words in positive training set with repitition
List_Size_Positive = len(list_positive_word)
#Dictionary having the words : count 
Positive_Word_Dict = Counter(list_positive_word)
Denominator_Sum_Pos = sum(Positive_Word_Dict.values())

#List of all unique words in Positive Training set
Unique_list_positive_word = set(list_positive_word)
#Length of Unique Size list
Unique_List_Size_Positive = len(Unique_list_positive_word)
Total_No_of_Docs = count_positive_ID + count_negative_ID
print ("count_positive_ID - ",count_positive_ID)
print ("count_negative_ID - ",count_negative_ID)
print ("Total_No_of_Docs - ",Total_No_of_Docs)

Log_Prior_Positive = np.log10(float(count_positive_ID)/float(Total_No_of_Docs))
Log_Prior_Negative = np.log10(float(count_negative_ID)/float(Total_No_of_Docs))
print ("Log_Prior_Positive - ",Log_Prior_Positive)
print ("Log_Prior_Negative - ",Log_Prior_Negative)

Negative_Word_Dict_final = defaultdict()
Positive_Word_Dict_final = defaultdict()

print ("Denominator_Sum_Pos",Denominator_Sum_Pos)
print ("Denominator_Sum_Neg",Denominator_Sum_Neg)
#Creating maximum likelihood estimate values for each word in positive training set
for key in Positive_Word_Dict:    
    val = np.log10 (float(Positive_Word_Dict[key] + 1)/float(Denominator_Sum_Pos + Unique_List_Size_Positive))
    Positive_Word_Dict_final.update({key:val})
for key in Negative_Word_Dict:    
    val =  np.log10(float(Negative_Word_Dict[key] + 1)/float(Denominator_Sum_Neg + Unique_List_Size_Negative))
    Negative_Word_Dict_final.update({key:val})

Negative_Probability_Sentence = 100.00
Positive_Probability_Sentence = 100.00
Sentence_Token_List = []
Sentence_List = []

with open("TestSet.txt",'r',encoding="utf8") as f:
   token =  [line.split() for line in f]
   for each_word in token:
        Sentence_Token_List.append(each_word)

Test_Review_Dict = OrderedDict()
Test_Review_Class = OrderedDict()
temp_dict = {}
List_Tokens_ID = []
temp = []

for i in range (0,len(Sentence_Token_List)):
    for j in range (0,len(Sentence_Token_List[i])):
        word = Sentence_Token_List[i][j]
        if re.match(r'ID-[0-9].*',word):
            Id_Value = word
            continue
        elif re.match(r'(\d+)\.(\d+)+',word):
            continue
        elif re.match(r'(\d+)+',word):
            continue 
        elif re.match(r'$(\d+)+',word):
            continue  
        elif re.match(r'[A-Za-z]*\.+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\?+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\!+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'-+',word):
            continue
        elif re.match(r'[A-Za-z]*\)+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'\(+[A-Za-z]*',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\,+',word):
            word = re.sub(r'[^\w\s]','',word)
        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',word):
            list_word = word.split('/')
            if list_word[0] in frequent_word_list:
                continue
            else:
                List_Tokens_ID.append(list_word[0])
            if list_word[1] in frequent_word_list:
                continue
            else:
                List_Tokens_ID.append(list_word[0])
                
        if (word not in frequent_word_list):
                List_Tokens_ID.append(word)
    temp = copy.copy(List_Tokens_ID)
    Test_Review_Dict.update ({Id_Value:temp})
    List_Tokens_ID[:] = []

for each_key,each_value in Test_Review_Dict.items():
    ID_Value = each_key
    for val in each_value:
        if ((val in Negative_Word_Dict_final) and (val in Positive_Word_Dict_final)):
            Negative_Probability_Sentence += Negative_Word_Dict_final[val]
            Positive_Probability_Sentence += Positive_Word_Dict_final[val]

    Positive_Probability_Sentence+=Log_Prior_Positive
    Negative_Probability_Sentence+=Log_Prior_Negative
    temp_dict.update({'POS': Positive_Probability_Sentence})
    temp_dict.update({'NEG': Negative_Probability_Sentence})
    print ("ID- Value : {}, Negative : {} , Positive : {}".format(ID_Value, Negative_Probability_Sentence,Positive_Probability_Sentence))
    key = [k for k,v in temp_dict.items() if v==max(temp_dict.values())][0] 
    Test_Review_Class.update ({ID_Value:key})
    temp_dict.clear()
    Positive_Probability_Sentence = 100.0
    Negative_Probability_Sentence = 100.0

    
with open("TestResult.txt",'w') as ofile:
    
    Positive_Probability_Sentence+=Log_Prior_Positive
    Negative_Probability_Sentence+=Log_Prior_Negative
    temp_dict.update({'POS': Positive_Probability_Sentence})
    temp_dict.update({'NEG': Negative_Probability_Sentence})
       key = [k for k,v in temp_dict.items() if v==max(temp_dict.values())][0] 
    Test_Review_Class.update ({ID_Value:key})
    temp_dict.clear()
    for keys,values in Test_Review_Class.items():
            ofile.write((str(keys) + '\t' + values + '\n'))
            ofile.write((str("ID- Value : {}, Negative : {} , Positive : {}".format(ID_Value, Negative_Probability_Sentence,Positive_Probability_Sentence))))
