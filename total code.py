# 저장된 데이터 파일 불러오기
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import scipy.io

mat_file_name="C:/Users/user/Documents/sEMG-TASK/sEMG.mat"
mat_file=scipy.io.loadmat(mat_file_name)
emg=mat_file['emg']
label=mat_file['label']
rep=mat_file['repetition']