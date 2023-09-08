# Bert-VITS2 Cook Book

## 1. 创建虚拟环境

```shell
virtualenv venv
source venv/bin/activate
```

## 2. 安装依赖项

```shell
pip install -r requirements.txt
```

## 3. 准备数据

1. 自定义数据集：

   * 若干条语音`path/to/dataset/<speaker_name>/*.wav`，可以放在任意路径下，因为会在`filelists/genshin.list`中注明；注意，语音文件的采样率必须与配置文件`configs/config.json`中的`data.sampling_rate`一致，如果不一致可以使用脚本`resample.py`进行重采样

   * 语音内容标注`filelist/genshin.list`（这个文件名是在脚本`preprocess_text.py`中写死的），其中每行的格式为：

     `{wav_path}|{speaker_name}|{language}|{text}`

     这是一个具体的例子：

     `/root/my_dataset/迪奥娜/114514.wav|迪奥娜|ZH|哼哼，快快开始激动人心的新人对局吧！`

2. 执行脚本`preprocess_text.py`对`filelist/genshin.list`做预处理：

   ```shell
   python preprocess_text.py
   ```

   这一步会在目录`filelists`生成中间文件`genshin.list.cleaned`以及用于模型训练评估的`train.list`和`val.list`

## 4. 准备底模

下载底模：[Stardust_minus/Bert-VITS2 - Bert-VITS2 - OpenI - 启智AI开源社区提供普惠算力！ (pcl.ac.cn)](https://openi.pcl.ac.cn/Stardust_minus/Bert-VITS2/modelmanage/model_filelist_tmpl?name=Bert-VITS2底模)

## 5. 下载BERT模型权重

在本repo中有目录`bert/chinese-roberta-wwm-ext-large`，这一目录对应的原始repo为[hfl/chinese-roberta-wwm-ext-large at main (huggingface.co)](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/tree/main)，可以完整克隆进行替换，也可以下载`pytorch_model.bin`放入目录`bert/chinese-roberta-wwm-ext-large`

## 6. 使用BERT生成prosody embedding

```shell
python bert_gen.py
```

这一步会在目录`path/to/dataset/<speaker_name>`中为每个音频生成`*.bert.pt`

## 7. 开始训练

```shell
python train_ms.py -c configs/config.json -m <path/to/base_models>
```

其中`<path/to/base_models>`为第四步中保存底模的路径。训练过程中默认每1000步保存一次checkpoint，保存路径为`logs/<base_models_folder_name>/*.pth`

## 8. Web UI，启动！

```shell
python webui.py --model logs/<base_models_folder_name>/G_xxx.pth
```

