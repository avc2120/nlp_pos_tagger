import sys
import nltk
import math
from nltk.corpus import brown as bn
from collections import  defaultdict    
from nltk.tokenize import word_tokenize
import itertools
#this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
	words = defaultdict(float)
	knownwords = []    
	for line in wbrown:
		for item in line:
			words[item] += 1
	for item in words:
		if words[item] > 5:
			knownwords.append(item)
    	return knownwords

#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    	rare = []
	for line in brown:
		sentence = []
		for word in line:
			if word not in knownwords:
				sentence.append('_RARE_')
			else:
				sentence.append(word)
		rare.append(sentence)
    	return rare

#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()

#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
	tokens = []
	bigram_c = defaultdict(float)
	trigram_c = defaultdict(float)
	trigram_p = defaultdict(float)

	for line in tbrown:
		tokens = ['*'] + line + ['STOP']
		bigram_tuples = (tuple(nltk.bigrams(tokens)))
		tokens = ['*'] + tokens
		trigram_tuples = (tuple(nltk.trigrams(tokens)))
		for pair in bigram_tuples:
			bigram_c[pair] += 1.0
		for triple in trigram_tuples:
			trigram_c[triple] += 1.0
		for key in trigram_c:
			if key[0] == '*' and key[1] == '*':
				trigram_p[key] = math.log(trigram_c[key],2).real - math.log(len(tbrown),2).real
			else:
				trigram_p[key] = math.log(trigram_c[key],2).real - math.log(bigram_c[(key[0],key[1])],2).real 
    	return trigram_p

#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()

#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):
    	evalues = defaultdict(float)
	tags = defaultdict(float)
	taglist = []
	for linew, linet in zip(wbrown, tbrown):
		for tag, word in zip(linet, linew):
			evalues[word, tag] +=1.0
			tags[tag] += 1.0
	for item in evalues:
		evalues[item] = math.log(evalues[item]/tags[item[1]],2)
	for item in tags:
		taglist.append(item)
    	return evalues, taglist

#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()

#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords),
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a
#sentence tagged in the WORD/TAG format
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues is from the return of calc_trigrams = probability of the trigrams of tags
#evalues is from the return of calc_emissions() = count(word, tag)/count(tag)
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string with a terminal newline, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
    	tagged = []
	pi = defaultdict(float)
	bp = {}
	pi[(-1,'*','*')] = 0.0
	for line in brown:
		tokens_orig =  nltk.word_tokenize(line)	
		tokens = [w if w in knownwords else '_RARE_' for w in tokens_orig]
		tokens = ['*'] + tokens + ['STOP']
		# k = 1 case
		for w in taglist:
			pi[(0, '*', w)] = pi[(-1,'*','*')] + qvalues.get(('*', '*', w), -1000) + evalues.get((tokens[0], w), -1000)
			bp[(0, '*', w)] = '*'

		# k = 2 case
		for (w, u) in itertools.product(taglist, taglist):
			key = ('*', w, u)
			pi[(1, w, u)] = pi.get((0, '*', w), -1000) + qvalues.get(key, -1000) + evalues.get((tokens[1], u), -1000)
			bp[(1, w, u)] = '*' 
		tags = []
		#k >= 2 case
		for k in range (2, len(tokens)):
			for (u, v) in itertools.product(taglist, taglist):
				max_prob = -float('Inf')
				max_tag = ""
				for w in taglist:
					score = pi.get((k-1, w, u), -1000) + qvalues.get((w,u,v), -1000) + evalues.get((tokens[k], v), -1000)
					if(score > max_prob):
						max_prob = score
						max_tag = w
				bp[(k,u,v)] = max_tag
				pi[(k,u,v)] = score
		
		max_prob = -float('Inf')
		#finding the max probability of last two tags
		for (u,v) in itertools.product(taglist,taglist):
			score = pi.get((len(tokens_orig)-1, u, v),-1000) + qvalues.get((u,v,'STOP'),-1000) 
			if score >  max_prob:
				max_prob = score
				u_max = u
				v_max = v

		#append tags in reverse order
		tags.append(v_max)
		tags.append(u_max)
		count = 0
		for k in range(len(tokens_orig)-3, -1, -1):
			tags.append(bp.get((k + 2, tags[count+1], tags[count]),-1000))
			count +=1
		tagged_sentence = ""
		#reverse tags
		tags.reverse()

		#stringify tags paired with word without start and stop symbols
		for k in range(0, len(tokens_orig)):
			tagged_sentence = tagged_sentence + tokens_orig[k] + "/" + str(tags[k]) + " "
		tagged_sentence += "\n"
		tagged.append(tagged_sentence)	
	return tagged		

#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    	outfile = open('B5.txt', 'w')
    	for sentence in tagged:
        	outfile.write(sentence)
    	outfile.close()

#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of tagged sentences the WORD/TAG format. Each sentence is a string with a terminal newline rather than a list of tokens.
def nltk_tagger(brown):
	tagged = []

	training = bn.tagged_sents(tagset = 'universal')
	default_tagger = nltk.DefaultTagger('NOUN')
	bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
	trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)
	for sentence in brown:
       		tagged_sentence = trigram_tagger.tag(sentence)
        # print sentence
        	sentence = [w + '/' + t for w, t in tagged_sentence]
        	tagged.append(' '.join(sentence) + '\n')	
	return tagged

def q6_output(tagged):
    	outfile = open('B6.txt', 'w')
	for sentence in tagged:
        	output = ' '.join(sentence) + '\n'
		outfile.write(output)
    	outfile.close()

#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only)
def split_wordtags(brown_train):
    	wbrown = []
    	tbrown = []
	for line in brown_train:
		tags = []
		words = []
		tokens = line.split()
		for token in tokens:
			word = token.rsplit('/', 1)
			for i in range(0, len(word)-1):
				words.append(word[i])
			tags += [word[len(word)-1]]
		wbrown.append(words)
		tbrown.append(tags)
	return wbrown, tbrown

def main():
    	#open Brown training data
    	infile = open("Brown_tagged_train.txt", "r")
    	brown_train = infile.readlines()
    	infile.close()
   	#split words and tags, and add start and stop symbols (question 1)
    	wbrown, tbrown = split_wordtags(brown_train)
	#calculate trigram probabilities (question 2)
    	qvalues = calc_trigrams(tbrown)
	print wbrown[0]
    	#question 2 output
    	q2_output(qvalues)

    	#calculate list of words with count > 5 (question 3)
    	knownwords = calc_known(wbrown)

    	#get a version of wbrown with rare words replace with '_RARE_' (question 3)
    	wbrown_rare = replace_rare(wbrown, knownwords)

    	#question 3 output
    	q3_output(wbrown_rare)

    	#calculate emission probabilities (question 4)
    	evalues, taglist = calc_emission(wbrown_rare, tbrown)
	
    	#question 4 output
    	q4_output(evalues)

    	#delete unneceessary data
    	del brown_train
    	del wbrown
    	del tbrown
    	del wbrown_rare
    	#open Brown development data (question 5)
    	infile = open("Brown_dev.txt", "r")
    	brown_dev = infile.readlines()
	infile.close()
	brown_dev = brown_dev[0:3]    
	#format Brown development data here
    	#do viterbi on brown_dev (question 5)
    	viterbi_tagged = viterbi(brown_dev, taglist, knownwords, qvalues, evalues)
	
    	#question 5 output
    	q5_output(viterbi_tagged)
	'''
    	#do nltk tagging here
    	nltk_tagged = nltk_tagger(brown_dev)

    	#question 6 output
    	q6_output(nltk_tagged)
	'''
if __name__ == "__main__": main()
