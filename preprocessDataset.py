import glob
import json
import os

import shutil

def safe_copy(source, destination):
    if os.path.isfile(destination):
        return 0
    
    try:
        shutil.copy2(source, destination)
    except:
        return 0
    return 1

path_VG100K = "/home/cmi/ThirdYearProject/dataset/VG_100K"
path_VG100K2 = "/home/cmi/ThirdYearProject/dataset/VG_100K_2"

images = glob.glob(path_VG100K + "/*.jpg")
temp = glob.glob(path_VG100K2 + "/*.jpg")
for image in temp:
    images.append(image)
print("Total: {}".format(len(images)))

with open('/home/cmi/ThirdYearProject/dataset/train_split.json', "r") as training:
    training_split = json.load(training)

with open('/home/cmi/ThirdYearProject/dataset/val_split.json', "r") as validation:
    validation_split = json.load(validation)

with open('/home/cmi/ThirdYearProject/dataset/test_split.json', "r") as test:
    testing_split = json.load(test)

print("Training: {}".format(len(training_split)))
print("Validation: {}".format(len(validation_split)))
print("Testing: {}".format(len(testing_split)))

total_n_images = len(training_split) + len(validation_split) + len(testing_split)

path_training = "/home/cmi/ThirdYearProject/dataset/training/"
path_validation = "/home/cmi/ThirdYearProject/dataset/validation/"
path_testing = "/home/cmi/ThirdYearProject/dataset/testing/"

count = 0
for index, image in enumerate(images):
    res = 0
    image_id = int(os.path.basename(image).split(".")[0])
    if image_id in training_split:
        res = safe_copy(image, os.path.join(path_training, os.path.basename(image)))
    elif image_id in validation_split:
        res = safe_copy(image, os.path.join(path_validation, os.path.basename(image)))
    elif image_id in testing_split:
        res = safe_copy(image, os.path.join(path_testing, os.path.basename(image)))
    
    if res != 0:
        count += 1
        print("{} / {}\r".format(count, total_n_images), end="", flush=True)
    else:
        print("{} - {}".format(index, image))
        # with open("/home/cmi/ThirdYearProject/dataset/errors.txt", "w") as f:
        #     f.write("{} - {}".format(index, image))

        

print()

training_set = glob.glob(path_training + "/*.jpg")
validation_set = glob.glob(path_validation + "/*.jpg")
testing_set = glob.glob(path_testing + "/*.jpg")

with open("/home/cmi/ThirdYearProject/dataset/training_set.txt", "w") as f:
    for image in training_set:
        f.write(image + "\n")

with open("/home/cmi/ThirdYearProject/dataset/validation_set.txt", "w") as f:
    for image in validation_set:
        f.write(image + "\n")

with open("/home/cmi/ThirdYearProject/dataset/testing_set.txt", "w") as f:
    for image in testing_set:
        f.write(image + "\n")
