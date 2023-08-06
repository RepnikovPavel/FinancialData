import os

data_folder = r'/home/user/FTC/ROW_DATA'
train_table = os.path.join(data_folder,'X_train.csv')
train_target = os.path.join(data_folder,'y_train.csv')
test_table = os.path.join(data_folder,'X_test.csv')

X_train_reformated = os.path.join(data_folder,'X_train_reformated.csv')
cat_encoders_path = os.path.join(data_folder,'cat_encoders.txt')
X_train_dataset = os.path.join(data_folder,'X_train_dataset.csv')
Y_train_dataset = os.path.join(data_folder,'Y_train_dataset.csv')


X_test_reformated = os.path.join(data_folder,'X_test_reformated.csv')
X_test_dataset = os.path.join(data_folder,'X_test_dataset.csv')


submission_csv = os.path.join(data_folder,'answer.csv')
submission_zip = os.path.join(data_folder,'answer.zip')

