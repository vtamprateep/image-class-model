import os, json, csv


def prepData( path, category):
    data = []

    for subdir, _, files in os.walk(path):
        label = subdir.split("/")[-1]
        if label not in category:
            continue

        for img in files:
            buffer = 9 - len(img)
            img_name = buffer * "0" + img
            data.append({"img_id" : img_name, "class" : label })
    return data

def exportCSV( data, set_type, save_path, columns=["img_id", "class"] ):
    
    file_name = save_path + set_type + "_data"
    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


image_categories = ['forest','buildings','glacier','street','mountain','sea']

### Training Data ###

train_data = prepData("./data_raw/seg_train/seg_train", image_categories)
exportCSV(train_data, "train", "./data_raw/")

test_data = prepData("./data_raw/seg_test/seg_test", image_categories)
exportCSV(test_data, "test", "./data_raw/")