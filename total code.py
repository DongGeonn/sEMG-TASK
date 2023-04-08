# 저장된 데이터 파일 불러오기
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import scipy.io

mat_file_name="C:/Users/user/Documents/sEMG-TASK/sEMG.mat"
mat_file=scipy.io.loadmat(mat_file_name)
emg=mat_file['emg']
label=mat_file['label']
rep=mat_file['repetition']

# data_first 빈 딕셔너리 만들기
data_first={}
for num_label in range(1,18):                                                          
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        data_first[key_name]=[]


# data 딕셔너리의 value에 입력값을 추가하는 함수 설정
def dicplus(dictionary, key, value):                                            
    value_list=dictionary[key]
    value_list.append(value)
    dictionary[key]=value_list


# data_first 딕셔너리의 각 key에 value 입력
for i in range(emg.shape[0]):                                                  
    for num_label in range(1,18):
        for num_rep in range(1,7):
            if num_label==label[i] and num_rep==rep[i]:
                key_name="L"+str(num_label)+"-"+str(num_rep)       
                dicplus(data_first, key_name, emg[i,:])


# data_first 딕셔너리의 각 valve를 numpy형태로 설정                
import numpy as np                  
for num_label in range(1,18):                                                   
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        data_first[key_name]=np.array(data_first[key_name])
        
# data_second 딕셔너리 생성 & 입력        
data_second={}
for num_label in range(1,18):                                                          
    for num_rep in range(1,7):
        key_name="L"+str(num_label)+"-"+str(num_rep)
        array=data_first[key_name]
        len_array=(array.shape[0]//400)*400
        array=array[:len_array,:]
        array=array.reshape((-1,400,12))
        data_second[key_name]=array