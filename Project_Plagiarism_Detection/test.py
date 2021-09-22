# # import libraries
# from sklearn.feature_extraction.text import CountVectorizer
# import sklearn
# import helpers
# import problem_unittests as tests
# import pandas as pd
# import numpy as np
# import os

# csv_file = 'data/file_information.csv'
# plagiarism_df = pd.read_csv(csv_file)

# # print out the first few rows of data info
# plagiarism_df.head()

# # Read in a csv file and return a transformed dataframe


# def numerical_dataframe(csv_file='data/file_information.csv'):
#     '''Reads in a csv file which is assumed to have `File`, `Category` and `Task` columns.
#         This function does two things:
#        1) converts `Category` column values to numerical values
#        2) Adds a new, numerical `Class` label column.
#        The `Class` column will label plagiarized answers as 1 and non-plagiarized as 0.
#        Source texts have a special label, -1.
#        :param csv_file: The directory for the file_information.csv file
#        :return: A dataframe with numerical categories and a new `Class` label column'''

#     # your code here
#     category_maps = {'non': 0, 'heavy': 1, 'light': 2, 'cut': 3, 'orig': -1}
#     class_maps = {'non': 0, 'heavy': 1, 'light': 1, 'cut': 1, 'orig': -1}

#     plagiarism_df = pd.read_csv(csv_file)
#     # Clone new col Class from Category column
#     plagiarism_df['Class'] = plagiarism_df['Category']
#     plagiarism_df['Category'].replace(category_maps, inplace=True)
#     plagiarism_df['Class'].replace(class_maps, inplace=True)

#     return plagiarism_df


# # informal testing, print out the results of a called function
# # create new `transformed_df`
# transformed_df = numerical_dataframe(csv_file='data/file_information.csv')

# # check work
# # check that all categories of plagiarism have a class label = 1
# transformed_df.head(10)

# # test cell that creates `transformed_df`, if tests are passed

# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """

# # importing tests

# # test numerical_dataframe function
# tests.test_numerical_df(numerical_dataframe)

# # if above test is passed, create NEW `transformed_df`
# transformed_df = numerical_dataframe(csv_file='data/file_information.csv')

# # check work
# print('\nExample data: ')
# transformed_df.head()


# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """

# # create a text column
# text_df = helpers.create_text_column(transformed_df)
# text_df.head()


# # after running the cell above
# # check out the processed text for a single file, by row index
# row_idx = 0  # feel free to change this index

# sample_text = text_df.iloc[0]['Text']

# print('Sample processed text:\n\n', sample_text)


# random_seed = 1  # can change; set for reproducibility

# """
# DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
# """

# # create new df with Datatype (train, test, orig) column
# # pass in `text_df` from above to create a complete dataframe, with all the information you need
# complete_df = helpers.train_test_dataframe(text_df, random_seed=random_seed)

# # check results
# complete_df.head(20)


# def containment(ngram_array):
#     ''' Containment is a measure of text similarity. It is the normalized, 
#        intersection of ngram word counts in two texts.
#        :param ngram_array: an array of ngram counts for an answer and source text.
#        :return: a normalized containment value.'''

#     # the intersection can be found by looking at the columns in the ngram array
#     # this creates a list that holds the min value found in a column
#     # so it will hold 0 if there are no matches, and 1+ for matching word(s)
#     intersection_list = np.amin(ngram_array, axis=0)

#     # optional debug: uncomment line below
#     # print(intersection_list)

#     # sum up number of the intersection counts
#     intersection = np.sum(intersection_list)

#     # count up the number of n-grams in the answer text
#     answer_idx = 0
#     answer_cnt = np.sum(ngram_array[answer_idx])

#     # normalize and get final containment value
#     containment_val = intersection / answer_cnt

#     return containment_val


# # Calculate the ngram containment for one answer file/source file pair in a df


# def calculate_containment(df, n, answer_filename):
#     '''Calculates the containment between a given answer text and its associated source text.
#        This function creates a count of ngrams (of a size, n) for each text file in our data.
#        Then calculates the containment by finding the ngram count for a given answer text, 
#        and its associated source text, and calculating the normalized intersection of those counts.
#        :param df: A dataframe with columns,
#            'File', 'Task', 'Category', 'Class', 'Text', and 'Datatype'
#        :param n: An integer that defines the ngram size
#        :param answer_filename: A filename for an answer text in the df, ex. 'g0pB_taskd.txt'
#        :return: A single containment value that represents the similarity
#            between an answer text and its source text.
#     '''

#     # your code here
    
#     answer_text = df.loc[df['File'] == answer_filename, 'Text']
#     print(answer_text)
#     answer_category = df.loc[df['File'] == answer_filename, 'Task']
#     print("Type:  ", type(answer_category.iloc[0]))
#     # print(df.loc[(df['Task'] == 'a')])
#     source_text = df.loc[(df['Task'] == answer_category.iloc[0])
#                          & (df['Datatype'] == 'orig'), 'Text']
#     print(source_text)

#     # instantiate an ngram counter
#     counts = CountVectorizer(analyzer='word', ngram_range=(n, n))

#     # create array of n-gram counts for the answer and source text
#     ngrams = counts.fit_transform([answer_text.iloc[0], source_text.iloc[0]])

#     return containment(ngrams.toarray())


# # select a value for n
# n = 3

# # indices for first few files
# test_indices = range(5)

# # iterate through files and calculate containment
# category_vals = []
# containment_vals = []
# for i in test_indices:
#     # get level of plagiarism for a given file index
#     category_vals.append(complete_df.loc[i, 'Category'])
#     # calculate containment for given file and n
#     filename = complete_df.loc[i, 'File']
#     c = calculate_containment(complete_df, n, filename)
#     containment_vals.append(c)

# # print out result, does it make sense?
# print('Original category values: \n', category_vals)
# print()
# print(str(n)+'-gram containment values: \n', containment_vals)

import pandas as pd

df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],

     index=['cobra', 'viper', 'sidewinder'],

     columns=['max_speed', 'shield'])

print(df)

temp = df.loc['cobra', 'shield']

print(temp)
print(type(temp))

temp = df.loc[df['shield'] > 6, ['max_speed']]

print(temp)
print(type(temp))