A4: Perplexity is a measurement of how well a probability distribution or probability model predicts a sample In this case, bettermodels of the unknown distributions tend to have lower perplexities due to higher x(xi) in the equation b^(-1/N)sum(q(xi)). Thus the model with linear interpolation drastically outperforms the model without for unigrams and bigrams. However for trigrams, the model without interpolation outperforms the model with interpolation. The perplexities are as follows:
A2.uni.txt: 1104.83292814
A2.bi.txt: 57.2215464238
A2.tri.txt: 5.89521267642
A3.txt: 13.076
In A2, the trigrams are more accurate than bigrams which are more accurate than unigrams as more information is present in trigrams leading to better tags. The perplexity of A2.tri is less than A3.txt because all the trigrams were found since the training data and the test were the same sentences. This means that the trigrams do a better job at tagging than linear interpolation because the linear interpolation takes in the weighted average of unigrams, bigrams, and trigrams.

A5: The perplexities run on Sample1_scored.txt and Sample2_scored.txt are as follows?
Sample1: The perplexity is 11.6492786046
Sample2: The perplexity is 1627571078.54
Since the perplexity for Sample1 is closer than Sample2, this means that Sample1 belongs in the data set
