# Import the necessary packages
import os  # allows us to navigate file structure
import pandas as pd  # allows us to import and manipulate data frames
from sklearn.preprocessing import LabelEncoder  # Label string values like sentence, bias, etc to numeric values
import numpy as np  # helps with linear algebra tasks
import pickle  # used to save python objects
import seaborn as sns  # used for plotting a heatmap of the data
import matplotlib.pyplot as plt


def main(path):
	def import_and_analyze_data(path, filename=None):
		df = None  # Bookkeeping
		encoder = LabelEncoder()  # also bookkeeping
		for i, file in enumerate(os.listdir(path)):  # i=0,1,2... file=names of files in path folder
			df1 = pd.read_excel(os.path.join(path, file), header=None)  # load in data
			df1 = df1.iloc[3:, 3:-1]  # select only the region we care about
			df1['AnnotatorID'] = [i]*len(df1)  # add annotatorID column so we know where data comes from
			if df is None:  # will only be None at beginning -- Can't concatenate data frames if one doesn't exist!
				df = df1  # set df (which was None) to our new imported dataset
			else:
				df = pd.concat([df, df1], axis=0)  # On the next run, df will not be None, and we begin merging datasets

		df.index = list(range(len(df)))  # change the index to be a numbered list from 0 to the length of the data
		df1 = pd.read_excel(os.path.join(path, os.listdir(path)[0]), header=None)
		columns = df1.iloc[1, 3:]  # select the column names from the original data (the new data cut out this part)
		columns[1:9] = df1.iloc[2, 4:-4]  # the excel is set up weird... the column names in the middle are a line below others
		columns[15] = 'AnnotatorID'  # Could have grabbed this from df1 too, but I was too lazy to find where it got put
		df.columns = columns  # replace data frame column names with the list of column names we just made
		cols = ['Citation', 'Claim Stance', 'Bias', 'Sentence']  # define columns we want to encode (makes computation faster)
		for col in cols:
			encoder = LabelEncoder()  # define an instantiation of the encoder
			df[col] = encoder.fit_transform(df[col].astype('str'))  # use .fit_transform method to encode the values and store
		# Get names of indexes for which all columns are empty
		cols = ['Events', 'Regulations', 'Quantity', 'Prediction', 'Personal', 'Normative', 'Other', 'No Claim']
		indexNames = df[(df[cols[0]].astype('str') == 'nan') & (df[cols[1]].astype('str') == 'nan') &
						(df[cols[2]].astype('str') == 'nan') & (df[cols[3]].astype('str') == 'nan') &
						(df[cols[4]].astype('str') == 'nan') & (df[cols[5]].astype('str') == 'nan') &
						(df[cols[6]].astype('str') == 'nan') & (df[cols[7]].astype('str') == 'nan')].index

		# Delete these row indexes from dataFrame
		df.drop(indexNames, inplace=True)
		# Get names of indexes for which all columns are full
		indexNames = df[(df[cols[0]].astype('str') == 'X') & (df[cols[1]].astype('str') == 'X') &
						(df[cols[2]].astype('str') == 'X') & (df[cols[3]].astype('str') == 'X') &
						(df[cols[4]].astype('str') == 'X') & (df[cols[5]].astype('str') == 'X') &
						(df[cols[6]].astype('str') == 'X') & (df[cols[7]].astype('str') == 'X')].index

		# Delete these row indexes from dataFrame
		df.drop(indexNames, inplace=True)

		# We want numeric values for faster processing, so we will map X to 1 and empty values to 0
		mapping = {'X': 1, 'nan': 0}
		for col in cols:
			df[col] = df[col].astype('str')  # .astype('str') method converts our selection's data type to string
			df[col] = df[col].map(mapping)

		# We need to define a function that depends on whether the surveyor responded yes or no, and then calculated
		# how many agreements there for that sentence for that column across all annotators

		def sentence_scorecard(x, sentence, col):
			if x == 1:
				return sum(df[df['Sentence'].astype('int32') == int(sentence)][col])/len(df[df['Sentence'].astype('int32') == int(sentence)])
			else:
				return (len(df[df['Sentence'].astype('int32') == int(sentence)]) - sum(df[df['Sentence'].astype('int32') == int(sentence)][col]))/len(df[df['Sentence'].astype('int32') == int(sentence)])
		# We now apply this function to each cell value using lambda x function
		for col in cols:
			df[col] = df.apply(lambda x: sentence_scorecard(x[col], x['Sentence'], col), axis=1)

		# There columns are more tricky because their value can be more than just 0 or 1
		# So, we find the total number that agree with our surveyor and divide by total # of responses
		cols = ['Citation', 'Claim Stance', 'Bias']

		def sentence_scorecard2(x, sentence, col):
			return sum(df[df['Sentence'] == int(sentence)][col].astype('int') == int(x))/len(df[df['Sentence'] == int(sentence)])

		# and we apply that function with a lambda x function
		for col in cols:
			df[col] = df.apply(lambda x: sentence_scorecard2(x[col], x['Sentence'], col), axis=1)

		# df['Sentence'] = encoder.inverse_transform(df['Sentence'])
		if filename is not None:
			df.to_excel(filename)
			with open(filename + '.pkl', 'wb') as file:
				pickle.dump(file, encoder)
		return df

	df = import_and_analyze_data(path=path)  # use the above function to get the sentence by column level agreement table
	new_df = pd.DataFrame()  # initialize an empty dataframe we will add to
	cols = df.columns.tolist()[1:-1] # select the columns of interest

	# we need a function that takes in the number of a sentence and a column, and finds the average agreement for all
	# annotators for that column
	def get_sentence_avg(sentence, col):
		df2 = df[df['Sentence'] == int(sentence)][col]
		return np.mean(df2)

	# we also need one that does this for annotators
	def get_annotator_avg(annotator, col):
		df2 = df[df['AnnotatorID'] == int(annotator)][col]
		return np.mean(df2)


	for sentence in set(df['Sentence']):  # loop through all possible sentences
		sentence_avgs = []  # every sentence, we want to start a new row
		for col in cols:  # now we loop through columns and use above function to find average
			sentence_avg = get_sentence_avg(sentence, col)  # do that ^
			sentence_avgs.append(sentence_avg)  # store this in order, so when we add our column names, they match
		new_df = new_df.append(pd.Series([sentence] + sentence_avgs), ignore_index=True)  # add the averages as a row to
	# our data frame, ignore_index just allows us to append without explicitly defining an index
	new_df.columns = ['Sentence'] + cols  # we need to add sentences to match how we defined this above
	new_df.to_excel('Sentence Averages.xlsx', index=False)  # save the data
	annotator_df = pd.DataFrame()  # now we'll do the same thing, but instead of looking at every sentence, we will look at
	# every annotator
	for annotator in set(df['AnnotatorID']):  # loop through possible annotators
		annotator_avgs = []  # initialize empty row vector for our data
		for col in cols:  # now loop through columns
			annotator_avgs.append(get_annotator_avg(annotator, col))  # add each column level average to our ordered list
		annotator_df = annotator_df.append(pd.Series([annotator] + annotator_avgs), ignore_index=True)  # after looping through
		# all columns, we append the whole row vector to our dataframe, including the sentence the values came from
	annotator_df.columns = ['AnnotatorID'] + cols  # need to add annotator ID to match how we defined row above
	annotator_df.to_excel('Annotator Averages.xlsx', index=False)  # save the data
	# The rest of the code is just so we can visualize the data and save the plots
	plt.tight_layout()
	heatmap = sns.heatmap(annotator_df[annotator_df.columns.tolist()[1:]].T)
	heatmap.set_xlabel('Annotator ID')
	plt.gcf().subplots_adjust(left=0.18)
	fig = heatmap.get_figure()
	fig.savefig('Heatmap of Annotator Agreement.png')
	plt.close()
	plt.tight_layout()
	heatmap = sns.heatmap(new_df[new_df.columns.tolist()[1:]].T)
	heatmap.set_xlabel('Sentence ID')
	plt.gcf().subplots_adjust(left=0.18)
	fig = heatmap.get_figure()
	fig.savefig('Heatmap of Sentence Agreement.png')

if __name__ == '__main__':
	path = './fake-news-master/week1.finished/'  # Define the path to a folder with our excel files in it
	main(path)  # run the function we defined that does all our computation/saving


# If at any point, you want to convert sentences from their numeric value to their true value...
# use encoder.inverse_transform(df['Sentence'])
