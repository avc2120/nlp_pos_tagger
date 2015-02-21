import math as m
import nltk
import cmath as math
from collections import defaultdict
#a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
def calc_probabilities(brown):
	tokens = []
	unigram_p = defaultdict(float)
	unigram_c = defaultdict(float)
	bigram_p  = defaultdict(float)
	trigram_p = defaultdict(float)
	trigram_c = defaultdict(float)
	bigram_c = defaultdict(float)
	for line in brown:
		tokens = (nltk.word_tokenize(line))
		tokens += ['STOP']
		for word in tokens:
			unigram_c[(word,)] += 1.0
		tokens = ['*'] + tokens
		bigram_tuples = (tuple(nltk.bigrams(tokens)))
		tokens = ['*'] + tokens
		trigram_tuples = (tuple(nltk.trigrams(tokens)))
		for pair in bigram_tuples:
			bigram_c[pair] += 1.0
		for triple in trigram_tuples:
			trigram_c[triple] += 1.0
	length = sum(unigram_c.values())		
	for  key in unigram_c:
		unigram_p[key] = math.log(unigram_c[key]/length,2).real			
	for key in bigram_c:
		if key[0] == '*':
			bigram_p[key] = math.log(bigram_c[key]/unigram_c[('STOP',)],2).real
		else:	
			bigram_p[key] = math.log(bigram_c[key]/unigram_c[(key[0],)],2).real
	for keys in trigram_c:
		if keys[0] == '*' and keys[1] == '*':
			trigram_p[keys] = math.log(trigram_c[keys],2).real - math.log(unigram_c[('STOP',)],2).real
		else:	
			trigram_p[keys] = math.log(trigram_c[keys],2).real- math.log(bigram_c[(keys[0], keys[1])],2).real
		
	return unigram_p, bigram_p, trigram_p

#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    outfile.close()
    
#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):
    	#ngram is a dictionary of probabilities of the ngram, n is number of grams, data is brown txt
	scores = []
	count = 0
	for line in data:
		tokens = []
		line_score = 0.0
		for i in range(3-n,2):
			tokens += ['*'] 
		tokens = tokens + (nltk.word_tokenize(line))
		tokens = tokens + ['STOP']
			
		for i in range(0,len(tokens)-n+1):
			key = ()
			for j in range(i,i+n):
				key += (tokens[j],)
			if ngram_p[key]!= 0.0:
				line_score += ngram_p[key]
			else:
				line_score = -1000
				break
		scores.append(line_score)
	return scores


#this function outputs the score output of score()
#scores is a python list of scores, and filename is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


#this function scores brown data with a linearly interpolated model
#each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
#like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, brown):
    	scores = []
	for line in brown:
		uni = []
		bi = []
		tri = []
		tokens = (nltk.word_tokenize(line)) + ['STOP']
		not_found = False
		for i in range(1,4):
			if i == 2 or i == 3:
				tokens = ['*'] + tokens
			for k in range (0, len(tokens)-i+1):
				key = ()
				for j in range(k, k+i):
					key+= (tokens[j],)
				if i == 1:
					if unigrams[key] == 0:
						not_found = True
						break
					uni.append(unigrams[key])
				if i == 2:
					if bigrams[key] == 0:
						not_found = True
						break
					bi.append(bigrams[key])
				if i == 3:
					if trigrams[key] == 0:
						not_found = True
						break
					tri.append(trigrams[key])

		score = 0.0
		if not_found:
			score = -1000
		else:
			for i in range(0, len(uni)):
				score += math.log(1.0/3.0*(m.pow(2, uni[i]) + m.pow(2,bi[i]) + m.pow(2,tri[i])),2)
		scores.append(score.real)		
    	return scores

def main():
    	#open data
    	infile = open('Brown_train.txt', 'r')
	brown = infile.readlines()
    	infile.close()
    	#calculate ngram probabilities (question 1)
    	unigrams, bigrams, trigrams = calc_probabilities(brown)

    	#question 1 output
    	q1_output(unigrams, bigrams, trigrams)

	#score sentences (question 2)
   	uniscores = score(unigrams, 1, brown)
    	biscores = score(bigrams, 2, brown)
    	triscores = score(trigrams, 3, brown)

    	#question 2 output
    	score_output(uniscores, 'A2.uni.txt')
    	score_output(biscores, 'A2.bi.txt')
    	score_output(triscores, 'A2.tri.txt')

   	#linear interpolation (question 3)
 	linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    	#question 3 output
    	score_output(linearscores, 'A3.txt')

	#open Sample1 and Sample2 (question 5)
    	infile = open('Sample1.txt', 'r')
    	sample1 = infile.readlines()
    	infile.close()
    	infile = open('Sample2.txt', 'r')
    	sample2 = infile.readlines()
    	infile.close() 

    	#score the samples
   	sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    	sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    	#question 5 output
    	score_output(sample1scores, 'Sample1_scored.txt')
    	score_output(sample2scores, 'Sample2_scored.txt')

if __name__ == "__main__": main()
