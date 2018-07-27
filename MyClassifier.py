import csv, math, sys


train_data = []
test_data = []

#euclodian distance between two records
def record_distance(a, b):
    sum = 0
    for x in range(0,8):
        sum += math.pow(a[x]-b[x], 2)
    return math.sqrt(sum)

#knn
def knn(t, k):
    def sort_fn(s):
        return record_distance(t, s)
    #this is slowwwww
    s = sorted(train_data, key=sort_fn)
    yes_count = 0
    for x in range(0,k):
        if s[x][8] == 1:
            yes_count += 1
    if yes_count >= k/2.0:
        return 1
    return 0

#preprocess train data csv - convert values to floats
with open(sys.argv[1]) as myFile:  
    reader = csv.reader(myFile)
    for row in reader:
        new = []
        if len(row) is not 9:
            continue
        for x in range(0,8):
            new.append(float(row[x]))
        if row[8] == "yes":
            new.append(1)
        elif row[8] == "no":
            new.append(0)
        train_data.append(new)

#preprocess test csv - convert values to floats
with open(sys.argv[2]) as myFile:  
    reader = csv.reader(myFile)
    for row in reader:
        new = []
        if len(row) < 8:
            continue
        for x in range(0,8):
            new.append(float(row[x]))
        test_data.append(new)


#runn NN
if sys.argv[3][-2:] == 'NN':
    for x in test_data:
        d = knn(x, int(sys.argv[3][0]))
        if d == 1:
            print("yes")
        else:
            print("no")

#calculate mean and std. dev of each attribute in the training set for use in naive bayes
def gen_train_stats(b):
    mean = []
    dev = []
    for x in range(0, 8):
        mean.append(sum((c[x] if c[8] == b else 0) for c in train_data)/float(sum(True if c[8] == b else False for c in train_data)))
        dev.append(sum(map(lambda a: pow(a[x] - mean[x], 2) if a[8] ==  b else 0, train_data))/float(sum(True if c[8] == b else False for c in train_data)))
    return mean, dev


#runn NB (with stats)
if sys.argv[3][-2:] == 'NB':
    train_mean_0, train_dev_0 = gen_train_stats(0)
    train_mean_1, train_dev_1 = gen_train_stats(1)
    p_0 = sum(True if c[8] == 0 else False for c in train_data)/float(len(train_data))
    p_1 = sum(True if c[8] == 1 else False for c in train_data)/float(len(train_data))
    def prob_dens(a, mean, dev):
        return 1/(math.sqrt(2*math.pi*dev)) * math.exp(-1*math.pow(a-mean, 2)/(2*dev))
    def nb_test(a):
        p_a_0 = 1
        for x in range(0, 8):
            p_a_0 *= prob_dens(a[x], train_mean_0[x], train_dev_0[x])
        p_a_1 = 1
        for x in range(0, 8):
            p_a_1 *= prob_dens(a[x], train_mean_1[x], train_dev_1[x])
        return p_a_1*p_1/(p_a_1*p_1+p_a_0*p_0)
    for x in test_data:
        if nb_test(x) < 0.5:
            print("no")
        else:
            print("yes")


#runn (with stats)
# if sys.argv[3][-2:] == 'NN':
#     correct = 0
#     for x in test_data:
#         d = knn(x, int(sys.argv[3][0]))
#         if len(x) > 8:
#             if d == int(x[8]):
#                 correct += 1
#         if d == 1:
#             print("yes")
#         else:
#             print("no")
#     print(str(correct) + " / " + str(len(test_data)))
#     print(str(correct/float(len(test_data))*100) + "%")