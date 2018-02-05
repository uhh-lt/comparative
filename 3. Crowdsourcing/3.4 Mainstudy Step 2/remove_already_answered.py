import pandas as pd


# brands data
input_data = pd.read_csv('data_before_removal_of_other/brands_(400)_1221734/source1221734.csv')
input_data = input_data[input_data['label_gold'].isnull()]
answered = pd.read_csv('data_before_removal_of_other/brands_(400)_1221734/a1221734.csv')

assert len(answered)  == 650
assert len(input_data)  == 2401

unanswered = pd.concat([input_data,answered]).drop_duplicates(keep=False,subset='id')
assert len(unanswered) == len(input_data) - len(answered)
check = unanswered[['id']].isin(answered['id'].values)
assert len(check['id'].value_counts()) == 1
assert check['id'].value_counts()[0] == len(input_data) - len(answered)
assert check['id'].value_counts().get(True) is None

unanswered[['a','also_acceptable','b','expected_label','id','marker','raw_text','text_html','text_readable']].to_csv('unanswered_brands.csv', index=False)

# compsci data
test_questions = pd.read_csv('compsci_test_questions.csv')
input_data = pd.read_csv('data_before_removal_of_other/compsci_1222555/source1222555.csv')
input_data = input_data[input_data['label_gold'].isnull()]
answered = pd.read_csv('data_before_removal_of_other/compsci_1222555/a1222555.csv')

assert len(answered)  == 750
assert len(input_data)  == 2500

unanswered = pd.concat([input_data,answered]).drop_duplicates(keep=False,subset='id')
unanswered = unanswered[~unanswered['id'].isin(list(test_questions['id'].unique()))]
check = unanswered[['id']].isin(answered['id'].values)
assert len(check['id'].value_counts()) == 1
assert check['id'].value_counts().get(True) is None

unanswered[['a','also_acceptable','b','expected_label','id','marker','raw_text','text_html','text_readable']].to_csv('unanswered_compsci.csv', index=False)
