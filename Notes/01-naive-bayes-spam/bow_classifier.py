from abstract_classifier import AbstractSPAMClassifier
import math


class BagOfWordsClassifier(AbstractSPAMClassifier):
    def fit(self, messages, labels):
        # Compute prior probabilities P(c), aka the probability of spam messages
        #计算垃圾邮件的总数
        message_count = messages.shape[0]
        total_ham_messages = labels.value_counts()[0]
        total_spam_messages = labels.value_counts()[1]
        total_messages = total_ham_messages + total_spam_messages
        self.log_prior_prob_ham = math.log(
            total_ham_messages / total_messages)
        self.log_prior_prob_spam = math.log(
            total_spam_messages / total_messages)

        # Split words (bigrams) into ham and spam classes for all messages
        #计算垃圾邮件和非垃圾邮件的条数以及集合
        self.ham_total, self.spam_total = 0, 0
        ham_words, spam_words = [], []
        for i in range(message_count):
            if labels[i] == 0:
                ham_words += messages[i]
                self.ham_total += 1
            else:
                spam_words += messages[i]
                self.spam_total += 1

        # Record word frequencies within each class
        #计算每个单词在垃圾邮件中和非垃圾邮件中出现的频数
        ham_freqs, spam_freqs = {}, {}
        for word in ham_words:
            ham_freqs[word] = ham_freqs.get(word, 0) + 1
        for word in spam_words:
            spam_freqs[word] = spam_freqs.get(word, 0) + 1

        # Compute the condtionl probabilites P(w|c)
        # 计算条件概率
        self.log_ham_prob, self.log_spam_prob = {}, {}
        for word, count in ham_freqs.items():
            self.log_ham_prob[word] = math.log(
                (count + 1) / (self.ham_total + 1))
        for word, count in spam_freqs.items():
            self.log_spam_prob[word] = math.log(
                (count + 1) / (self.spam_total + 1))

    def predict(self, message):
        prob_ham = self.log_prior_prob_ham
        prob_spam = self.log_prior_prob_spam
        for word in message:
            if word in self.log_ham_prob:
                prob_ham += self.log_ham_prob[word]
            if word in self.log_spam_prob:
                prob_spam += self.log_spam_prob[word]
            if word not in self.log_ham_prob and word not in self.log_spam_prob:
                prob_ham += math.log(1 / self.ham_total )
        return int(prob_spam < prob_ham)
