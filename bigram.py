import pandas as pd
import numpy as np
import math

# Load datasets.
data = pd.read_csv("./SMSSpamCollection", sep='\t', header=None)

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o','p', 'q', 'r', 's', 't', 'u',
 'v', 'w', 'x','y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '/', '?', '!', '.', 'α', 'β', 'γ']

total_num_sms = len(data[0])

# preprocess the dataset
def data_preprocessing(sms):
    # convert to lowercase.
    sms = sms.lower()

    # replace all unknow character to 'γ', where 'γ' is
    # taken as replacement for unknown character.
    sms = "".join([ c if c in alphabet else 'γ' for c in sms ])

    # append start ('α') and end ('β') symbols.
    sms = 'α' + sms + 'β'

    return sms

for i in range(total_num_sms):
    data[1][i] = data_preprocessing(data[1][i])

# determine min and max indices of train data.
train_min_index = 0
train_max_index = int((total_num_sms * 80) / 100)

# determine min and max indices of test data.
test_min_index = train_max_index + 1
test_max_index = total_num_sms

# determine number of classes.
num_classes = len(np.unique(data[0]))

alphabet_count_spam = dict()
alphabet_count_ham = dict()

# count alphabet characters in ham and spam training dataset.
def count_trainingset_alphabet_spam_ham():
    # first set items in alphabet to 0.
    for char in alphabet:
        alphabet_count_spam[char] = 0
        alphabet_count_ham[char] = 0

    # count each item of alphabet in training data.
    for i in range(train_min_index, train_max_index):
        sms = data[1][i]
        if data[0][i] == "spam":
            for c in sms:
                alphabet_count_spam[c] = alphabet_count_spam[c] + 1
        else:
            for c in sms:
                alphabet_count_ham[c] = alphabet_count_ham[c] + 1

count_trainingset_alphabet_spam_ham()

bigram_count_spam = dict()
bigram_count_ham = dict()

# count character bigram in ham and spam training dataset.
def count_trainingset_bigram_spam_ham():
    # first set bigram count to 0.
    for first_char in alphabet:
        for second_char in alphabet:
            bigram = first_char + second_char
            bigram_count_spam[bigram] = 0
            bigram_count_ham[bigram] = 0

    # count each bigram in training data.
    for i in range(train_min_index, train_max_index):
        sms = data[1][i]
        if data[0][i] == "spam":
            for char in range(len(sms) - 1):
                bigram = sms[char] + sms[char + 1]
                bigram_count_spam[bigram] = bigram_count_spam[bigram] + 1
        else:
            for char in range(len(sms) - 1):
                bigram = sms[char] + sms[char + 1]
                bigram_count_ham[bigram] = bigram_count_ham[bigram] + 1

count_trainingset_bigram_spam_ham()

# taking k as 0.5 for finding prediction.
k = 0.5

prob_bigram_ham = dict()
prob_bigram_spam = dict()

# determine probabilities of bigrams in training dataset.
def prob_bigram_spam_ham():
    for bigram, count in bigram_count_spam.items():
        bigram_first_char = bigram[0]
        prob_bigram_spam[bigram] = (count + k) / (alphabet_count_spam[bigram_first_char] + (k * len(alphabet)))

    for bigram, count in bigram_count_ham.items():
        bigram_first_char = bigram[0]
        prob_bigram_ham[bigram] = (count + k) / (alphabet_count_ham[bigram_first_char] + (k * len(alphabet)))

prob_bigram_spam_ham()

# determine log probability of a test SMS message.
# used log to eliminate arithmatic undeflow.
def log_probability(x, spam_ham):
    log_prob = 0
    if spam_ham == "spam":
        for char in range(len(x) - 1):
            bigram = x[char] + x[char + 1]
            log_prob = log_prob + math.log(prob_bigram_spam[bigram])
    else:
        for char in range(len(x) - 1):
            bigram = x[char] + x[char + 1]
            log_prob = log_prob + math.log(prob_bigram_ham[bigram])

    return log_prob

# predict the test SMS message for ham or spam.
def predict(x):
    spam_predict = log_probability(x, "spam")
    ham_predict = log_probability(x, "ham")
    
    if spam_predict >= ham_predict:
        return "spam"
    else:
        return "ham"

# predict test SMS message in each row in the test dataset.
test_data_count = test_max_index - test_min_index
predicted_classes = np.empty((test_data_count), dtype="<U10")
for n in range(test_min_index, test_max_index):
    index = n - test_min_index
    predicted_classes[index] = predict(data[1][n])

print("predicted classes of test SMS messages")
print(predicted_classes)

# determine the confusion matrix.
def confusion_matrix(predicted_classes, actual_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for i in range(len(predicted_classes)):
        if predicted_classes[i] == "spam":
            predicted_class = 0
        else:
            predicted_class = 1

        if actual_classes[i] == "spam":
            actual_class = 0
        else:
            actual_class = 1

        cm[predicted_class][actual_class] = cm[predicted_class][actual_class] + 1

    return cm

# determine precision macroaverage.
def p_macroaverage():
    p_macro = np.zeros((num_classes), dtype=np.float64)
    
    for i in range(num_classes):
        sum_t = 0 
        frac = confusion_matrix[i][i]
        for j in range(num_classes):
            sum_t = sum_t + confusion_matrix[j][i]

        p_macro[i] = frac / sum_t

    return np.sum(p_macro) / num_classes

# determine recall macroaverage.
def r_macroaverage():
    r_macro = np.zeros((num_classes), dtype=np.float64)

    for i in range(num_classes):
        sum_t = 0 
        frac = confusion_matrix[i][i]
        for j in range(num_classes):
            sum_t = sum_t + confusion_matrix[i][j]

        r_macro[i] = frac / sum_t

    return np.sum(r_macro) / num_classes


actual_classes = np.array(data[0])[test_min_index:test_max_index]
print("actual classes of test SMS messages")
print(actual_classes)

# determine accuracy of prediction compared to the actual values.
accuracy = 0
for i in range(len(predicted_classes)):
    if predicted_classes[i] == actual_classes[i]:
        accuracy = accuracy + 1

accuracy = accuracy / len(predicted_classes)
print("accuracy: ", accuracy)

confusion_matrix = confusion_matrix(predicted_classes, actual_classes)
print(confusion_matrix)

p_m = p_macroaverage()
r_m = r_macroaverage()
print("p_macroaverage: ", p_m)
print("r_macroaverage: ", r_m)

# determine the F1-score.
f1_score = (2 * p_m * r_m) / (p_m + r_m) 
print("F1-score: ", f1_score)


# predict spam or ham for following sms:
spam_msg = "Save up to Rs 150 this Valentine's Day ! "
ham_msg = "We recommend that you change your password immediately to keep your account secure."

# predict if spam message is predicted to be spam
pre_processd_sms = data_preprocessing(spam_msg)
prediction = predict(pre_processd_sms)
print("message: \"", spam_msg, "\" is : ", prediction)

# predict if ham message is predicted to be ham
pre_processd_sms = data_preprocessing(ham_msg)
prediction = predict(pre_processd_sms)
print("message: \"", ham_msg, "\" is : ", prediction)
