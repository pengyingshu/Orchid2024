import os
import json


def save_json(category2index:dict, path:str):
    assert os.path.exists(path), f"{path} not found"
    data = {}
    dirs = os.listdir(path)
    dirs.sort()
    for index, category in enumerate(dirs):
        pcategory = os.path.join(path, category)
        for file in os.listdir(pcategory):
            pfile = os.path.join(pcategory, file)
            data[pfile] = category2index[category]

    with open("{}.json".format(path), "w", encoding='utf-8') as f:
        f.write(json.dumps(data))


def main(dataset_path:str):

    category2index = {}
    os.chdir(dataset_path)

    ptrain = "train"
    assert os.path.exists(ptrain), f"{ptrain} not found"
    info_train = os.listdir(ptrain)
    info_train.sort()
    for index, category in enumerate(info_train):
        category2index[category] = index
    with open(f"category2index.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(category2index))

    save_json(category2index, "train")
    save_json(category2index, "val")
    save_json(category2index, "test")


if __name__ == '__main__':
    dataset_path = "datasets/orchid2024"
    main(dataset_path)
