from nltk.corpus import names
from nltk.classify import NaiveBayesClassifier as nbc, accuracy
from random import shuffle

males = names.raw('male.txt').split('\n')
females = names.raw('female.txt').split('\n')

shuffle(males)
shuffle(females)

males_data = [({'name':m}, 'male') for m in males[:2500]]
females_data = [({'name':f}, 'female') for f in females[:2500]]

train_data = males_data[:2000] + females_data[:2000]
test_data = males_data[2000:] + females_data[2000:]

classifier = nbc.train(train_data)
print('Percentage: ', accuracy(classifier, test_data) * 100, '%')

print(classifier.most_informative_features(5))

name = ''
while True:
    name = input("Insert name: ")
    result = classifier.classify({'name':name})
    print(result)