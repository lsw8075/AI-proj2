import hmm
import os
import time
import random

def calc_match(data1, data2):
    count = 0
    for (item1, item2) in zip(data1, data2):
        if item1 == item2:
            count += 1
    return count

def mysplit(s):
    x = s.split('/')
    return (x[1].lower(), x[0].lower())

def fine_split(sl):
    res = []
    for s in sl:
        tail = []
        if len(s) > 0 and s[0] == "\"":
            res.append("\"")
            s = s[1:]
        if len(s) > 0 and s[0] == "\'":
            res.append("\'")
            s = s[1:]
        if len(s) > 1 and s[0:1] == "--":
            res.append("--")
            s = s[2:]
        if len(s) > 0 and s[-1] == "\"":
            tail.append("\"")
            s = s[:-1]
        if len(s) > 0 and s[-1] == "\'":
            tail.append("\'")
            s = s[:-1]
        if len(s) > 1 and s[0:1] == "--":
            tail.append("--")
            s = s[:-2]
        for char in "!?,.":
            if len(s) > 0 and s[-1] == char:
                tail.append(char)
                s = s[:-1]
        tail.reverse()
        if len(s) > 0:
            res.append(s)
        res.extend(tail)

    return res

def op_train(filelist):
    print("Train file list:", filelist)
    datas = []
    i = 0
    for train_file in filelist:
        with open('brown/' + train_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                splited = line.split()
                data = [mysplit(s) for s in splited]
                datas.append(list(zip(*data)))
        i += 1
            
    print("data count:", str(len(datas)))

    starttime = time.time()
    model = hmm.train(datas)
    endtime = time.time()
    print("Training finished:", endtime - starttime, "s elapsed")

    model.dump('trained2')

def op_count():
    model = hmm.Model()
    model.load('trained')

    print("There are", \
        len(model._states), "states,", \
        len(model._symbols), "symbols,")
    
    print(model._states)
    print(model._symbols)
    print(model._start_prob)
    print(model._trans_prob)
    print(model._emit_prob)

def op_test(filelist):
    print("Test file list:", filelist)
    print(len(filelist), "files total")

    model = hmm.Model()
    model.load('trained')

    datas = []
    answers = []
    i = 0
    for test_file in filelist:
        with open('brown/' + test_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                splited = line.split()
                splited2 = [s.split('/') for s in splited]
                data = [s[0].lower() for s in splited2]
                answer = [s[1].lower() for s in splited2]
                datas.append(data)
                answers.append(answer)
        i += 1

    testtime = 0
    total_match = 0
    total_datanum = 0
    for i, (data, answer) in enumerate(zip(datas, answers)): 
        print("test data", i)
        print(data)
        starttime = time.time()
        prob = model.evaluate(data)
        result = model.decode(data)
        endtime = time.time()
        testtime += endtime - starttime
        print("Prob:", prob)
        print("result:", result)
        match = calc_match(result, answer)
        datanum = len(data)
        total_match += match
        total_datanum += datanum
        print("accuracy", match / datanum)
        print("=============================")
    print("Total test time:", testtime, "s")
    print("Total accuracy", total_match / total_datanum)

def main():
    print("[select operation]")
    print("1. train")
    print("2. test")
    print("3. train and test")
    print("4. check the model")
    print("otherwise. exit")

    sel = 0
    try:
        sel = int(input())
    except:
        return

    filelist = os.listdir('brown')

    train_list = []
    test_list = []
    
    train_count = 100
    test_count = 1

    for _ in range(train_count):
        s = random.choice(filelist)
        train_list.append(s)
        filelist.remove(s)
    
    for _ in range(test_count):
        s = random.choice(filelist)
        test_list.append(s)
        filelist.remove(s)

    if sel == 1:
        op_train(train_list)
    elif sel == 2:
        op_test(test_list)
    elif sel == 3:
        op_train(train_list)
        op_test(test_list)
    elif sel == 4:
        op_count()
    
if __name__ == '__main__':
    main()