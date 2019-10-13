from abstract_classifier import AbstractSPAMClassifier
import math


class BagOfWordsClassifier(AbstractSPAMClassifier):
    def fit(self, messages, labels):
        # Compute prior probabilities P(c), aka the probability of spam messages
        message_count = messages.shape[0]
        class1_messages = labels.value_counts()[0]
        class2_messages = labels.value_counts()[1]
        class3_messages = labels.value_counts()[2]
        self.py1 = math.log(
            class1_messages / message_count)
        self.py2 = math.log(
            class2_messages / message_count)
        self.py3 = math.log(
            class3_messages / message_count)

        # Split words (bigrams) into ham and spam classes for all messages
        #计算垃圾邮件和非垃圾邮件的条数以及集合
        class1_total, class2_total, class3_total = 0, 0, 0
        class1_words1, class1_words2, class1_words3, class1_words4 = [], [], [], []
        class2_words1, class2_words2, class2_words3, class2_words4 = [], [], [], []
        class3_words1, class3_words2, class3_words3, class3_words4 = [], [], [], []
        for i in range(message_count):
            if labels[i] == '0':
                class1_words1.append(messages[i][0])
                class1_words2.append(messages[i][1])
                class1_words3.append(messages[i][2])
                class1_words4.append(messages[i][3])
                class1_total += 1
            elif labels[i] == '1':
                class2_words1.append(messages[i][0])
                class2_words2.append(messages[i][1])
                class2_words3.append(messages[i][2])
                class2_words4.append(messages[i][3])
                class2_total += 1
            else:
                class3_words1.append(messages[i][0])
                class3_words2.append(messages[i][1])
                class3_words3.append(messages[i][2])
                class3_words4.append(messages[i][3])
                class3_total += 1

        # Record word frequencies within each class
        #计算每个单词在垃圾邮件中和非垃圾邮件中出现的频数
        class1_freqs1, class1_freqs2, class1_freqs3, class1_freqs4 = {}, {}, {}, {}
        class2_freqs1, class2_freqs2, class2_freqs3, class2_freqs4 = {}, {}, {}, {}
        class3_freqs1, class3_freqs2, class3_freqs3, class3_freqs4 = {}, {}, {}, {}
        for word in class1_words1:
            class1_freqs1[word] = class1_freqs1.get(word, 0) + 1
        for word in class1_words2:
            class1_freqs2[word] = class1_freqs2.get(word, 0) + 1
        for word in class1_words3:
            class1_freqs3[word] = class1_freqs3.get(word, 0) + 1
        for word in class1_words4:
            class1_freqs4[word] = class1_freqs4.get(word, 0) + 1
        for word in class2_words1:
            class2_freqs1[word] = class2_freqs1.get(word, 0) + 1
        for word in class2_words2:
            class2_freqs2[word] = class2_freqs2.get(word, 0) + 1
        for word in class2_words3:
            class2_freqs3[word] = class2_freqs3.get(word, 0) + 1
        for word in class2_words4:
            class2_freqs4[word] = class2_freqs4.get(word, 0) + 1
        for word in class3_words1:
            class3_freqs1[word] = class3_freqs1.get(word, 0) + 1
        for word in class3_words2:
            class3_freqs2[word] = class3_freqs2.get(word, 0) + 1
        for word in class3_words3:
            class3_freqs3[word] = class3_freqs3.get(word, 0) + 1
        for word in class3_words4:
            class3_freqs4[word] = class3_freqs4.get(word, 0) + 1

        # Compute the condtionl probabilites P(w|c)
        # 计算条件概率
        self.log_class1_prob1, self.log_class1_prob2, self.log_class1_prob3, self.log_class1_prob4 = {}, {}, {}, {}
        self.log_class2_prob1, self.log_class2_prob2, self.log_class2_prob3, self.log_class2_prob4 = {}, {}, {}, {}
        self.log_class3_prob1, self.log_class3_prob2, self.log_class3_prob3, self.log_class3_prob4 = {}, {}, {}, {}
        for word, count in class1_freqs1.items():
            self.log_class1_prob1[word] = math.log(
                (count + 1) / (class1_total + 1))
        for word, count in class1_freqs2.items():
            self.log_class1_prob2[word] = math.log(
                (count + 1) / (class1_total + 1))
        for word, count in class1_freqs3.items():
            self.log_class1_prob3[word] = math.log(
                (count + 1) / (class1_total + 1))
        for word, count in class1_freqs4.items():
            self.log_class1_prob4[word] = math.log(
                (count + 1) / (class1_total + 1))
        for word, count in class2_freqs1.items():
            self.log_class1_prob1[word] = math.log(
                (count + 1) / (class2_total + 1))
        for word, count in class2_freqs2.items():
            self.log_class1_prob2[word] = math.log(
                (count + 1) / (class2_total + 1))
        for word, count in class2_freqs3.items():
            self.log_class1_prob3[word] = math.log(
                (count + 1) / (class2_total + 1))
        for word, count in class2_freqs4.items():
            self.log_class1_prob4[word] = math.log(
                (count + 1) / (class2_total + 1))
        for word, count in class3_freqs1.items():
            self.log_class1_prob1[word] = math.log(
                (count + 1) / (class3_total + 1))
        for word, count in class3_freqs2.items():
            self.log_class1_prob2[word] = math.log(
                (count + 1) / (class3_total + 1))
        for word, count in class3_freqs3.items():
            self.log_class1_prob3[word] = math.log(
                (count + 1) / (class3_total + 1))
        for word, count in class3_freqs4.items():
            self.log_class1_prob4[word] = math.log(
                (count + 1) / (class3_total + 1))

    def predict(self, message):
        prob_class1 = self.py1
        prob_class2 = self.py2
        prob_class3 = self.py3
        
        word0 = message[0]
        if word0 in self.log_class1_prob1:
            prob_class1 += self.log_class1_prob1[word0]
        if word0 in self.log_class2_prob1:
            prob_class2 += self.log_class2_prob1[word0]
        if word0 in self.log_class3_prob1:
            prob_class3 += self.log_class3_prob1[word0]

        word1 = message[1]
        if word1 in self.log_class1_prob2:
            prob_class1 += self.log_class1_prob2[word1]
        if word1 in self.log_class2_prob2:
            prob_class2 += self.log_class2_prob2[word1]
        if word1 in self.log_class3_prob2:
            prob_class3 += self.log_class3_prob2[word1]

        word2 = message[2]
        if word2 in self.log_class1_prob3:
            prob_class1 += self.log_class1_prob3[word2]
        if word2 in self.log_class2_prob3:
            prob_class2 += self.log_class2_prob3[word2]
        if word2 in self.log_class3_prob3:
            prob_class3 += self.log_class3_prob3[word2]

        word3 = message[3]
        if word3 in self.log_class1_prob4:
            prob_class1 += self.log_class1_prob4[word3]
        if word3 in self.log_class2_prob4:
            prob_class2 += self.log_class2_prob4[word3]
        if word3 in self.log_class3_prob1:
            prob_class3 += self.log_class3_prob4[word3]
        
        p1 = prob_class1
        p2 = prob_class2
        p3 = prob_class3
        #print(p1, p2, p3)

        if p1 > p2:
            if p1 > p3:
                result = 0
            else:
                result = 2
        else:
            if p2 > p3:
                result = 1
            else:
                result = 2
        return result