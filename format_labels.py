import json

CLASSES = "data\\meta\\classes.txt"
LABELS_PATH = "data\\meta\\labels.txt"
TRAIN_DATA = "data\\meta\\train.json"
TESTING_DATA = "data\\meta\\test.json"
FORMATED_TRAIN = "data\\train.csv"
FORMATED_TEST = "data\\test.csv"
CLASS_DICT = "data\\classdict.json"
LABEL_DICT = "data\\labeldict.json"

with open(CLASSES, "r") as c, open(LABELS_PATH, "r") as l:
    classes = c.readlines()
    labels = l.readlines()

    if len(classes) != len(labels):
        raise Exception("Mismatching number of classes and labels")

    class_dict = {classes[i].strip(): i for i in range(len(classes))}
    label_dict = {i: labels[i].strip() for i in range(len(classes))}

    with open(CLASS_DICT, "w") as f:
        json.dump(class_dict, f)
    with open(LABEL_DICT, "w") as f:
        json.dump(label_dict, f)


with open(TRAIN_DATA, "r") as t:
    data = json.load(t)

    with open(FORMATED_TRAIN, "w") as f:
        f.write("Label, File_path\n")
        for food in data:
            for img in data[food]:
                f.write(str(class_dict[food]) + "," + img.replace("/", "\\") + ".jpg\n")

with open(TESTING_DATA, "r") as t:
    data = json.load(t)

    with open(FORMATED_TEST, "w") as f:
        f.write("Label, File_path\n")
        for food in data:
            for img in data[food]:
                f.write(str(class_dict[food]) + "," + img.replace("/", "\\") + ".jpg\n")
