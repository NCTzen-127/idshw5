from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import math 

domainlist1= [] 
domainlist2= []
class Domain:
	def __init__(self,_name,_label):
		self.name = _name
		self.label = _label

	def returnData(self):
	        return[len(self.name),countnum(self.name),cal_entropy(self.name)]

	def returnLabel(self):
		if self.label == "dga":
			return 0
		else:
			return 1

def countnum(string):
        int_count=0
        for i in string:
                if i.isdigit():
                        int_count +=1

        return int_count

def cal_entropy(str):
    h = 0.0
    sumLetter = 0
    letter = [0] * 26
    str = str.lower()
    for i in range(len(str)):
        if str[i].isalpha():
            letter[ord(str[i]) - ord('a')] += 1
            sumLetter += 1
    for i in range(26):
        p = 1.0 * letter[i] / sumLetter
        if p > 0:
            h += -(p * math.log(p, 2))
    return h

def initData1(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			name = tokens[0]
			label = tokens[1]
			domainlist1.append(Domain(name,label))

def initData2(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			tokens = line.split(",")
			name = tokens[0]
			label = "?"
			domainlist2.append(Domain(name,label))

def main():
	initData1("train.txt")
	initData2("test.txt")
	featureMatrix = []
	labelList = []
	for item in domainlist1:
		featureMatrix.append(item.returnData())
		labelList.append(item.returnLabel())

	clf = RandomForestClassifier(random_state=0)
	clf.fit(featureMatrix,labelList)
	
	arr=["dga","notdga"]
	f=open("result.txt",'w')
	for item in domainlist2:
		t=clf.predict([item.returnData()])
		f.write(item.name+','+np.array(arr)[t][0]+'\n')
		                        

if __name__ == '__main__':
	main()
