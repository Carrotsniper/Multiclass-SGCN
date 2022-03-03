labels=["Biker","Pedestrian","Car","Bus","Skater","Cart"]

one_hot_encoding = {}
for i in range(len(labels)):
    encoding = [0.] * len(labels)
    encoding[len(labels) - 1 - i] = 1.
    one_hot_encoding[labels[i]] = encoding
