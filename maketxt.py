import os

classes = {
    'InteractingWithComputer':0, 
    'Photographing':1, 
    'PlayingMusic':2, 
    'RidingBike':3, 
    'RidingHorse':4, 
    'Running':5, 
    'Walking':6
    }

d = {}
for root, dirs, files in os.walk('D:\\data\\Willow\\willowactions\\JPEGImages'):
    print(len(files))
    for file in files:
        d[file] = 0

# print(d)

s = set()
for root, dirs, files in os.walk('D:\\data\\Willow\\willowactions\\ImageSets\\Action'):
    for file in files:
        if 'trainval' in file or 'test' in file:
            label = classes[file.split('_')[0]]
            print(os.path.join(root, file))
            f = open(os.path.join(root, file))
            lines = f.readlines()
            f.close()
            for line in lines:
                if not d.__contains__(line.split()[0]):
                    d[line.split()[0]] = set()
                if line.split()[-1] == '1':
                    d[line.split()[0]].add(label)

for root, dirs, files in os.walk('D:\\data\\Willow\\willowactions\\ImageSets\\Action'):
    for file in files:
        if 'trainval' in file:
            f = open(os.path.join(root, file))
            lines = f.readlines()
            f.close()
            s = set()
            for line in lines:
                s.add(line.split()[0])
            f = open('D:\\data\\Willow\\willowactions\\trainval.txt', 'w')
            for img in s:
                f.write(img+' '+str(list(d[img])[0])+'\n')
            f.close()
            print('trainval', len(s))
        # if 'test' in file:
        #     f = open(os.path.join(root, file))
        #     lines = f.readlines()
        #     f.close()
        #     s = set()
        #     for line in lines:
        #         s.add(line.split()[0])
        #     f = open('D:\\data\\Willow\\willowactions\\test.txt', 'w')
        #     for img in s:
        #         f.write(img+' '+str(list(d[img])[0])+'\n')
        #     f.close()
        #     print('test', len(s))
            break
        