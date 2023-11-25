from sklearn.cluster import *
import os
import numpy as np
from config import config
import yaml
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--algorithm", default="k", help="choose algorithm", type=str
    )
    parser.add_argument(
        "-n", "--num_clusters", default=3, help="number of clusters", type=int
    )
    parser.add_argument(
        "-r", "--range", default=4, help="number of files in a class", type=int
    )
    args = parser.parse_args()
    filelist_dict = {}
    yml_result = {}
    with open(
        config.preprocess_text_config.cleaned_path, mode="r", encoding="utf-8"
    ) as f:
        embs = []
        wavnames = []
        for line in f:
            speaker = line.split("|")[1]
            if speaker not in filelist_dict:
                filelist_dict[speaker] = []
                yml_result[speaker] = {}
            filelist_dict[speaker].append(line.split("|")[0])
    # print(filelist_dict)

    for speaker in filelist_dict:
        print("\nspeaker: " + speaker)
        for file in filelist_dict[speaker]:
            try:
                embs.append(
                    np.expand_dims(
                        np.load(f"{os.path.splitext(file)[0]}.emo.npy"), axis=0
                    )
                )
                wavnames.append(os.path.basename(file))
            except Exception as e:
                print(e)
        x = np.concatenate(embs, axis=0)
        x = np.squeeze(x)
        # 聚类算法类的数量
        n_clusters = args.num_clusters
        if args.algorithm == "b":
            model = Birch(n_clusters=n_clusters, threshold=0.2)
        elif args.algorithm == "s":
            model = SpectralClustering(n_clusters=n_clusters)
        elif args.algorithm == "a":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=10)
        # 可以自行尝试各种不同的聚类算法
        y_predict = model.fit_predict(x)
        classes = [[] for i in range(y_predict.max() + 1)]

        for idx, wavname in enumerate(wavnames):
            classes[y_predict[idx]].append(wavname)

        for i in range(y_predict.max() + 1):
            print("类别:", i, "本类中样本数量:", len(classes[i]))
            yml_result[speaker][f"class{i}"] = []
            for j in range(args.range):
                print(classes[i][j])
                yml_result[speaker][f"class{i}"].append(classes[i][j])
    with open(
        os.path.join(config.dataset_path, "emo_clustering.yml"), "w", encoding="utf-8"
    ) as f:
        yaml.dump(yml_result, f)
