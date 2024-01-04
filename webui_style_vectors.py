import json
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE

from common.constants import DEFAULT_STYLE
from common.log import logger
from config import config

MAX_CLUSTER_NUM = 10
MAX_AUDIO_NUM = 10

tsne = TSNE(n_components=2, random_state=42, metric="cosine")


wav_files = []
x = np.array([])
x_tsne = None
mean = np.array([])
centroids = []


def load(model_name):
    global wav_files, x, x_tsne, mean
    wavs_dir = os.path.join("Data", model_name, "wavs")
    style_vector_files = [
        os.path.join(wavs_dir, f) for f in os.listdir(wavs_dir) if f.endswith(".npy")
    ]
    wav_files = [f.replace(".npy", "") for f in style_vector_files]
    style_vectors = [np.load(f) for f in style_vector_files]
    x = np.array(style_vectors)
    mean = np.mean(x, axis=0)

    x_tsne = tsne.fit_transform(x)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    return plt


def do_clustering(n_clusters=4, method="KMeans"):
    global centroids, x_tsne
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x)
    elif method == "KMeans after t-SNE":
        if x_tsne is None:
            x_tsne = tsne.fit_transform(x)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x_tsne)
    elif method == "Agglomerative after t-SNE":
        if x_tsne is None:
            x_tsne = tsne.fit_transform(x)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x_tsne)
    else:
        raise ValueError("Invalid method")

    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))

    return y_pred, centroids


def do_dbscan(eps=2.5, min_samples=15):
    global centroids, x_tsne
    model = DBSCAN(eps=eps, min_samples=min_samples)
    if x_tsne is None:
        x_tsne = tsne.fit_transform(x)
    y_pred = model.fit_predict(x_tsne)
    n_clusters = max(y_pred) + 1
    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))
    return y_pred, centroids


def closest_wav_files(cluster_index, num_files=1, weight=5):
    # centroidを強調した点からの距離が最も近い音声を選ぶ
    centroid_enhanced = mean + weight * (centroids - mean)
    # セントロイドと全ての点との距離を計算
    distances = cdist(centroid_enhanced[cluster_index : cluster_index + 1], x)
    # 距離が小さい順にソートし、上位のインデックスを取得
    closest_indices = np.argsort(distances[0])[:num_files]

    return closest_indices


def do_dbscan_gradio(eps=2.5, min_samples=15):
    global x_tsne, centroids

    y_pred, centroids = do_dbscan(eps, min_samples)

    if x_tsne is None:
        x_tsne = tsne.fit_transform(x)

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(max(y_pred) + 1):
        plt.scatter(
            x_tsne[y_pred == i, 0],
            x_tsne[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    # Noise cluster (-1) is black
    plt.scatter(
        x_tsne[y_pred == -1, 0],
        x_tsne[y_pred == -1, 1],
        color="black",
        label="Noise",
    )
    plt.legend()

    n_clusters = max(y_pred) + 1

    if n_clusters > MAX_CLUSTER_NUM:
        # raise ValueError(f"The number of clusters is too large: {n_clusters}")
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            f"クラスタ数が多すぎます、パラメータを変えてみてください。: {n_clusters}",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    elif n_clusters == 0:
        return [
            plt,
            gr.Slider(maximum=MAX_CLUSTER_NUM),
            f"クラスタが数が0です。パラメータを変えてみてください。",
        ] + [gr.Audio(visible=False)] * MAX_AUDIO_NUM

    return [plt, gr.Slider(maximum=n_clusters, value=1), n_clusters] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def closest_wav_files_gradio(cluster_index, num_files=1, weight=1):
    cluster_index = cluster_index - 1  # UIでは1から始まるので0からにする
    closest_indices = closest_wav_files(cluster_index, num_files, weight)
    return [
        gr.Audio(wav_files[i], visible=True, label=wav_files[i])
        for i in closest_indices
    ] + [gr.update(visible=False)] * (MAX_AUDIO_NUM - num_files)


def do_clustering_gradio(n_clusters=4, method="KMeans"):
    global x_tsne, centroids
    y_pred, centroids = do_clustering(n_clusters, method)

    if x_tsne is None:
        x_tsne = tsne.fit_transform(x)

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(n_clusters):
        plt.scatter(
            x_tsne[y_pred == i, 0],
            x_tsne[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i + 1}",
        )
    plt.legend()

    return [plt, gr.Slider(maximum=n_clusters, value=1)] + [
        gr.Audio(visible=False)
    ] * MAX_AUDIO_NUM


def save_style_vectors(model_name, style_names: str):
    """centerとcentroidsを保存する"""
    result_dir = os.path.join(config.assets_root, model_name)
    os.makedirs(result_dir, exist_ok=True)
    style_vectors = np.stack([mean] + centroids)
    style_vector_path = os.path.join(result_dir, "style_vectors.npy")
    np.save(style_vector_path, style_vectors)

    # config.jsonの更新
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        return f"{config_path}が存在しません。"
    style_name_list = [DEFAULT_STYLE]
    style_name_list = style_name_list + style_names.split(",")
    if len(style_name_list) != len(centroids) + 1:
        return f"スタイルの数が合いません。`,`で正しく{len(centroids)}個に区切られているか確認してください: {style_names}"
    style_name_list = [name.strip() for name in style_name_list]

    with open(config_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    return f"成功!\n{style_vector_path}に保存し{config_path}を更新しました。"


def save_style_vectors_from_files(model_name, audio_files_text, style_names_text):
    """音声ファイルからスタイルベクトルを作成して保存する"""
    global mean
    if len(x) == 0:
        return "Error: スタイルベクトルを読み込んでください。"
    mean = np.mean(x, axis=0)

    result_dir = os.path.join(config.assets_root, model_name)
    os.makedirs(result_dir, exist_ok=True)
    audio_files = audio_files_text.split(",")
    style_names = style_names_text.split(",")
    if len(audio_files) != len(style_names):
        return f"音声ファイルとスタイル名の数が合いません。`,`で正しく{len(style_names)}個に区切られているか確認してください: {audio_files_text}と{style_names_text}"
    audio_files = [name.strip() for name in audio_files]
    style_names = [name.strip() for name in style_names]
    style_vectors = [mean]

    wavs_dir = os.path.join("Data", model_name, "wavs")
    for audio_file in audio_files:
        path = os.path.join(wavs_dir, audio_file)
        if not os.path.exists(path):
            return f"{path}が存在しません。"
        style_vectors.append(np.load(f"{path}.npy"))
    style_vectors = np.stack(style_vectors)
    style_vector_path = os.path.join(result_dir, "style_vectors.npy")
    np.save(style_vector_path, style_vectors)

    # config.jsonの更新
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        return f"{config_path}が存在しません。"
    style_name_list = [DEFAULT_STYLE]
    style_name_list = style_name_list + style_names
    assert len(style_name_list) == len(style_vectors)

    with open(config_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    return f"成功!\n{style_vector_path}に保存し{config_path}を更新しました。"


initial_md = f"""
# Style Bert-VITS2 スタイルベクトルの作成

Style-Bert-VITS2でこまかくスタイルを指定して音声合成するには、モデルごとにスタイルベクトルのファイル`style_vectors.npy`を手動で作成する必要があります。

ただし、学習の過程で自動的に平均スタイル「{DEFAULT_STYLE}」のみは作成されるので、それをそのまま使うこともできます（その場合はこのWebUIは使いません）。

このプロセスは学習とは全く関係がないので、何回でも独立して繰り返して試せます。また学習中にもたぶん軽いので動くはずです。

## 方法

- 方法1: 音声ファイルを自動でスタイル別に分け、その各スタイルの平均を取って保存
- 方法2: スタイルを代表する音声ファイルを手動で選んで、その音声のスタイルベクトルを保存
- 方法3: 自分でもっと頑張ってこだわって作る（JVNVコーパスなど、もともとスタイルラベル等が利用可能な場合はこれがよいかも）
"""

method1 = """
学習の時に取り出したスタイルベクトルを読み込んで、可視化を見ながらスタイルを分けていきます。

手順:
1. 図を眺める
2. スタイル数を決める（平均スタイルを除く）
3. スタイル分けを行って結果を確認
4. スタイルの名前を決めて保存


詳細: スタイルベクトル(256次元)たちを適当なアルゴリズムでクラスタリングして、各クラスタの中心のベクトル（と全体の平均ベクトル）を保存します。

平均スタイル（{DEFAULT_EMOTION}）は自動的に保存されます。
"""

dbscan_md = """
DBSCANという方法でスタイル分けを行います。
こちらの方が方法1よりも特徴がより良く出るものができるかもしれません。

ただし事前にスタイル数は指定できません。

パラメータ：
- eps: この値より近い点同士をどんどん繋げて同じスタイル分類とする。小さいほどスタイル数が増え、大きいほどスタイル数が減る。
- min_samples: クラスタとみなすために必要な点の数。小さいほどスタイル数が増え、大きいほどスタイル数が減る。
"""

with gr.Blocks(theme="NoCrypt/miku") as app:
    gr.Markdown(initial_md)
    with gr.Row():
        model_name = gr.Textbox("your_model_name", label="モデル名")
        load_button = gr.Button("スタイルベクトルを読み込む", variant="primary")
    output = gr.Plot(label="音声スタイルの可視化")
    load_button.click(load, inputs=[model_name], outputs=[output])
    with gr.Tab("方法1: スタイル分けを自動で行う"):
        with gr.Tab("スタイル分け1"):
            n_clusters = gr.Slider(
                minimum=2,
                maximum=10,
                step=1,
                value=4,
                label="作るスタイルの数（平均スタイルを除く）",
                info="上の図を見ながらスタイルの数を試行錯誤してください。",
            )
            c_method = gr.Radio(
                choices=[
                    "Agglomerative after t-SNE",
                    "KMeans after t-SNE",
                    "Agglomerative",
                    "KMeans",
                ],
                label="アルゴリズム",
                info="分類する（クラスタリング）アルゴリズムを選択します。いろいろ試してみてください。",
                value="Agglomerative after t-SNE",
            )
            c_button = gr.Button("スタイル分けを実行")
        with gr.Tab("スタイル分け2: DBSCAN"):
            gr.Markdown(dbscan_md)
            eps = gr.Slider(
                minimum=0.1,
                maximum=10,
                step=0.1,
                value=2.5,
                label="eps",
                info="小さいほどスタイル数が増える",
            )
            min_samples = gr.Slider(
                minimum=1,
                maximum=100,
                step=1,
                value=15,
                label="min_samples",
                info="小さいほどスタイル数が増える",
            )
            with gr.Row():
                dbscan_button = gr.Button("スタイル分けを実行")
                num_styles_result = gr.Textbox(label="スタイル数")
        gr.Markdown("スタイル分けの結果")
        gr.Markdown("注意: もともと256次元なものをを2次元に落としているので、正確なベクトルの位置関係ではありません。")
        with gr.Row():
            gr_plot = gr.Plot()
            with gr.Column():
                with gr.Row():
                    cluster_index = gr.Slider(
                        minimum=1,
                        maximum=MAX_CLUSTER_NUM,
                        step=1,
                        value=1,
                        label="スタイル番号",
                        info="選択したスタイルの代表音声を表示します。",
                    )
                    num_files = gr.Slider(
                        minimum=1,
                        maximum=MAX_AUDIO_NUM,
                        step=1,
                        value=5,
                        label="代表音声の数をいくつ表示するか",
                    )
                    get_audios_button = gr.Button("代表音声を取得")
                with gr.Row():
                    audio_list = []
                    for i in range(MAX_AUDIO_NUM):
                        audio_list.append(
                            gr.Audio(visible=False, scale=1, show_label=True)
                        )
            c_button.click(
                do_clustering_gradio,
                inputs=[n_clusters, c_method],
                outputs=[gr_plot, cluster_index] + audio_list,
            )
            dbscan_button.click(
                do_dbscan_gradio,
                inputs=[eps, min_samples],
                outputs=[gr_plot, cluster_index, num_styles_result] + audio_list,
            )
            get_audios_button.click(
                closest_wav_files_gradio,
                inputs=[cluster_index, num_files],
                outputs=audio_list,
            )
        gr.Markdown("結果が良さそうなら、これを保存します。")
        style_names = gr.Textbox(
            "Angry, Sad, Happy",
            label="スタイルの名前",
            info=f"スタイルの名前を`,`で区切って入力してください（日本語可）。例: `Angry, Sad, Happy`や`怒り, 悲しみ, 喜び`など。平均音声は{DEFAULT_STYLE}として自動的に保存されます。",
        )
        with gr.Row():
            save_button = gr.Button("スタイルベクトルを保存", variant="primary")
            info2 = gr.Textbox(label="保存結果")

        save_button.click(
            save_style_vectors, inputs=[model_name, style_names], outputs=[info2]
        )
    with gr.Tab("方法2: 手動でスタイルを選ぶ"):
        gr.Markdown("下のテキスト欄に、各スタイルの代表音声のファイル名を`,`区切りで、その横に対応するスタイル名を`,`区切りで入力してください。")
        gr.Markdown("例: `angry.wav, sad.wav, happy.wav`と`Angry, Sad, Happy`")
        gr.Markdown(
            f"注意: {DEFAULT_STYLE}スタイルは自動的に保存されます、手動では{DEFAULT_STYLE}という名前のスタイルは指定しないでください。"
        )
        with gr.Row():
            audio_files_text = gr.Textbox(
                label="音声ファイル名", placeholder="angry.wav, sad.wav, happy.wav"
            )
            style_names_text = gr.Textbox(
                label="スタイル名", placeholder="Angry, Sad, Happy"
            )
        with gr.Row():
            save_button3 = gr.Button("スタイルベクトルを保存", variant="primary")
            info3 = gr.Textbox(label="保存結果")
            save_button3.click(
                save_style_vectors_from_files,
                inputs=[model_name, audio_files_text, style_names_text],
                outputs=[info3],
            )
        gr.Markdown("結果が良さそうなら、これを保存します。")
        style_names2 = gr.Textbox(
            "Angry, Sad, Happy",
            label="スタイルの名前",
            info=f"スタイルの名前を`,`で区切って入力してください（日本語可）。例: `Angry, Sad, Happy`や`怒り, 悲しみ, 喜び`など。平均音声は{DEFAULT_STYLE}として自動的に保存されます。",
        )
        with gr.Row():
            save_button2 = gr.Button("スタイルベクトルを保存", variant="primary")
            info4 = gr.Textbox(label="保存結果")

    app.launch(inbrowser=True)
