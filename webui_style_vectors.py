import json
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from config import config

MAX_CLUSTER_NUM = 10

tsne = TSNE(n_components=2, random_state=42, metric="cosine")


wav_files = []
x = np.array([])
x_tsne = None
mean = np.array([])
centroids = []


def load(model_name):
    global wav_files, x, x_tsne, mean
    wavs_dir = os.path.join("Data", model_name, "wavs")
    wav_files = [
        os.path.join(wavs_dir, wav_path)
        for wav_path in os.listdir(wavs_dir)
        if wav_path.endswith(".wav")
    ]
    style_vectors = [np.load(f"{wav_path}.npy") for wav_path in wav_files]
    x = np.array(style_vectors)
    mean = np.mean(x, axis=0)

    x_tsne = tsne.fit_transform(x)

    plt.figure(figsize=(6, 6))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    return plt


def do_clustering(n_clusters=4, method="KMeans"):
    global centroids
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x)
    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x)
    elif method == "KMeans after t-SNE":
        x_tsne = tsne.fit_transform(x)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        y_pred = model.fit_predict(x_tsne)
    elif method == "Agglomerative after t-SNE":
        x_tsne = tsne.fit_transform(x)
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit_predict(x_tsne)
    else:
        raise ValueError("Invalid method")

    centroids = []
    for i in range(n_clusters):
        centroids.append(np.mean(x[y_pred == i], axis=0))

    return y_pred, centroids


def closest_wav_files():
    # centroidを強調した点からの距離が最も近い音声を選ぶ
    centroid_enhanced = mean + 10 * (centroids - mean)
    closest_wav_files = []
    indices = []
    for i in range(len(centroids)):
        index = np.argmin(np.linalg.norm(x - centroid_enhanced[i], axis=1))
        indices.append(index)
        closest_wav_files.append(wav_files[index])
    return closest_wav_files, indices


def do_clustering_gradio(n_clusters=4, method="KMeans"):
    global x_tsne, centroids
    y_pred, centroids = do_clustering(n_clusters, method)
    representatives, indices = closest_wav_files()

    if x_tsne is None:
        x_tsne = tsne.fit_transform(x)

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6, 6))
    for i in range(n_clusters):
        plt.scatter(
            x_tsne[y_pred == i, 0],
            x_tsne[y_pred == i, 1],
            color=cmap(i),
            label=f"Style {i+1}",
        )
    plt.legend()

    plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], c="black", marker="x", s=100)

    return (
        [plt]
        + [gr.Audio(wav_path, visible=True) for wav_path in representatives]
        + [gr.update(visible=False)] * (MAX_CLUSTER_NUM - n_clusters)
        + [
            gr.Markdown(
                value=f"Style {i + 1}: {os.path.basename(wav_path)}", visible=True
            )
            for i, wav_path in enumerate(representatives)
        ]
        + [gr.update(visible=False)] * (MAX_CLUSTER_NUM - n_clusters)
    )


def save_style_vectors(model_name, style_names: str):
    """centerとcentroidsを保存する"""
    result_dir = os.path.join(config.out_dir, model_name)
    os.makedirs(result_dir, exist_ok=True)
    style_vectors = np.stack([mean] + centroids)
    style_vector_path = os.path.join(result_dir, "style_vectors.npy")
    if os.path.exists(style_vector_path):
        return f"{style_vector_path}が既に存在します。削除するか別の名前にバックアップしてください。"
    np.save(os.path.join(result_dir, "style_vectors.npy"), style_vectors)

    # config.jsonの更新
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.exists(config_path):
        return f"{config_path}が存在しません。"
    style_name_list = ["Neutral"]
    style_name_list = style_name_list + style_names.split(",")
    if len(style_name_list) != len(centroids) + 1:
        return f"スタイルの数が合いません。`,`で正しく{len(centroids)}個に区切られているか確認してください: {style_names}"
    style_name_list = [name.strip() for name in style_name_list]

    with open(config_path, "r") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(style_name_list)
    style_dict = {name: i for i, name in enumerate(style_name_list)}
    json_dict["data"]["style2id"] = style_dict
    with open(config_path, "w") as f:
        json.dump(json_dict, f, indent=2)
    return f"成功!\n{style_vector_path}に保存し{config_path}を更新しました。"


md1 = """
# Style Bert-VITS2 スタイルベクトルの作成

スタイルを使って音声合成するには、音声ファイルをスタイル別に分け、その各スタイルの特徴を抽出して保存する必要があります。

学習の時に取り出したスタイルベクトルを読み込んで、可視化を見ながらスタイルを分けていきます。

手順:
1. 図を眺める
2. スタイル数を決める（平均スタイルを除く）
3. スタイル分けを行って結果を確認
4. スタイルの名前を決めて保存

このプロセスは学習とは関係がないので、何回でも独立して繰り返して試せます。また学習中にもたぶん軽いので動くはずです。

詳細: スタイルベクトル(256次元)たちを適当なアルゴリズムでクラスタリングして、各クラスタの中心のベクトル（と全体の平均ベクトル）を保存します。

平均スタイル（Neutral）は自動的に保存されます。
"""

with gr.Blocks(theme="NoCrypt/miku") as app:
    gr.Markdown(md1)
    with gr.Row():
        model_name = gr.Textbox("your_model_name", label="モデル名")
        load_button = gr.Button("スタイルベクトルを読み込む", variant="primary")
    output = gr.Plot(label="音声スタイルの可視化")
    load_button.click(load, inputs=[model_name], outputs=[output])
    n_clusters = gr.Slider(
        minimum=2,
        maximum=10,
        step=1,
        value=4,
        label="作るスタイルの数（平均スタイルを除く）",
        info="上の図を見ながらスタイルの数を試行錯誤してください。",
    )
    c_method = gr.Radio(
        ["Agglomerative after t-SNE", "KMeans after t-SNE", "Agglomerative", "KMeans"],
        label="アルゴリズム",
        info="分類する（クラスタリング）アルゴリズムを選択します。いろいろ試してみてください。",
        value="Agglomerative after t-SNE",
    )
    c_button = gr.Button("スタイル分けを実行")
    audio_list = []
    md_list = []
    gr.Markdown("スタイル分けの結果と、各スタイルの特徴的な代表音声（図の黒い x 印）")
    gr.Markdown("注意: もともと256次元なものをを2次元に落としているので、正確なベクトルの位置関係ではありません。")
    with gr.Row():
        gr_plot = gr.Plot()
        with gr.Row():
            for i in range(MAX_CLUSTER_NUM):
                with gr.Column():
                    md_list.append(gr.Markdown(visible=False))
                    audio_list.append(
                        gr.Audio(
                            visible=False,
                            scale=1,
                            show_label=True,
                            label=f"スタイル{i+1}",
                        )
                    )
    c_button.click(
        do_clustering_gradio,
        inputs=[n_clusters, c_method],
        outputs=[gr_plot] + audio_list + md_list,
    )
    gr.Markdown("結果が良さそうなら、これを保存します。")
    with gr.Row():
        style_names = gr.Textbox(
            "Angry, Sad, Happy",
            label="スタイルの名前",
            info="スタイルの名前を`,`で区切って入力してください（日本語可）。例: `Angry, Sad, Happy`や`怒り, 悲しみ, 喜び`など。平均音声はNeutralとして自動的に保存されます。",
        )
        save_button = gr.Button("スタイルベクトルを保存", variant="primary")
        info = gr.Textbox(label="保存結果")

    save_button.click(
        save_style_vectors, inputs=[model_name, style_names], outputs=[info]
    )

app.launch(inbrowser=True)
