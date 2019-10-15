from abstract_classifier import AbstractIrisClassifier
import math


class IrisClassifier(AbstractIrisClassifier):
    def fit(self, data, target):
        # Compute prior probabilities P(c), aka the probability of spam messages
        #计算垃圾邮件的总数
        total_count = data.shape[0]
        self.total_class0, self.total_class1, self.total_class2 = 0, 0, 0
        for word in target:
            if(word == 0):
                self.total_class0 += 1
            if(word == 1):
                self.total_class1 += 1
            if(word == 2):
                self.total_class2 += 1
        
        self.log_prior_prob_class0 = math.log(
            self.total_class0 / total_count)
        self.log_prior_prob_class1 = math.log(
            self.total_class1 / total_count)
        self.log_prior_prob_class2 = math.log(
            self.total_class2 / total_count)

        # Split words (bigrams) into ham and spam classes for all messages
        #计算垃圾邮件和非垃圾邮件的条数以及集合
        class_feature = [
            [], [], [], [],
            [], [], [], [],
            [], [], [], []
        ]
        for i in range(total_count):
                for j in range(4):
                    class_feature[int(target[i])*4+j] += data[i][j]

        # Record word frequencies within each class
        #计算每个单词在垃圾邮件中和非垃圾邮件中出现的频数
        num_freqs = [
            {}, {}, {}, {},
            {}, {}, {}, {},
            {}, {}, {}, {}
        ]
        for i in range(3):
            for j in range(4):
                for word in class_feature[i*4+j]:
                    num_freqs[i*4+j][word] = num_freqs[i*4+j].get(word, 0) + 1

        # Compute the condtionl probabilites P(w|c)
        # 计算条件概率
        self.log_prob = [
            {}, {}, {}, {},
            {}, {}, {}, {},
            {}, {}, {}, {}
        ]
        for i in range(3):
            for j in range(4):
                for word, count in num_freqs[i*4+j].items():
                    self.log_prob[i*4+j][word] = math.log(
                        (count + 1) / (len(class_feature[i][j]) + 1))

    def predict(self, message):
        p1, p2, p3 = self.log_prior_prob_class0, self.log_prior_prob_class1, self.log_prior_prob_class2

        if(message[0] in self.log_prob[0]):
            p1 += self.log_prob[0][message[0]]
        else:
            p1 += math.log( 2 / self.total_class0 + 1)
        if(message[1] in self.log_prob[1]):
            p1 += self.log_prob[1][message[1]]
        else:
            p1 += math.log( 2 / self.total_class0 + 1)
        if(message[2] in self.log_prob[2]):
            p1 += self.log_prob[2][message[2]]
        else:
            p1 += math.log( 2 / self.total_class0 + 1)
        if(message[3] in self.log_prob[3]):
            p1 += self.log_prob[3][message[3]]
        else:
            p1 += math.log( 2 / self.total_class0 + 1)

        if(message[0] in self.log_prob[4]):
            p2 += self.log_prob[4][message[0]]
        else:
            p2 += math.log( 2 / self.total_class1 + 1)
        if(message[1] in self.log_prob[5]):
            p2 += self.log_prob[5][message[1]]
        else:
            p2 += math.log( 2 / self.total_class1 + 1)
        if(message[2] in self.log_prob[6]):
            p2 += self.log_prob[6][message[2]]
        else:
            p2 += math.log( 2 / self.total_class1 + 1)
        if(message[3] in self.log_prob[7]):
            p2 += self.log_prob[7][message[3]]
        else:
            p2 += math.log( 2 / self.total_class1 + 1)

        if(message[0] in self.log_prob[8]):
            p3 += self.log_prob[8][message[0]]
        else:
            p3 += math.log( 2 / self.total_class2 + 1)
        if(message[1] in self.log_prob[9]):
            p3 += self.log_prob[9][message[1]]
        else:
            p3 += math.log( 2 / self.total_class2 + 1)
        if(message[2] in self.log_prob[10]):
            p3 += self.log_prob[10][message[2]]
        else:
            p3 += math.log( 2 / self.total_class2 + 1)
        if(message[3] in self.log_prob[11]):
            p3 += self.log_prob[11][message[3]]
        else:
            p3 += math.log( 2 / self.total_class2 + 1)

        
        result = 0
        if (p1 > p2 and p1 > p3):
            result = 0
        if (p2 > p1 and p2 > p3):
            result = 1
        if (p3 > p1 and p3 > p2):
            result = 2
        return result
