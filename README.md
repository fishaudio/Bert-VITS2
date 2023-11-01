->进行标注（若已标注就跳过）-->预处理得到转录文件.list-->resample进行重采样 此时会得到dataset数据集
—>preprocess_text.py得到.clean文件(字符？) 以及划分训练集和数据集(同时会更新config文件中的spk2id键值，存储对应的人物信息)，—>—> bert_gen在每个人物语音数据集文件夹中生成pt文件 
—> train_ms.py生成总的混合模型（结合底模进行微调）

- 路径变了之后 需要改config.json 以及bert_gen.py之前的文件
- clean_raw 是为了整理列表和wav文件的，在这之前可以标注()完了之后 用clean_list去做个清理
- rename clean_list clean_raw 文件需要指定路径
- 注意python bert_gen.py 前要确保config中的train路径正确
- 训练混合角色模型时 数据集物理位置不需要变，只需要集成list即可（raw_list改完，重新clean raw一下 ，手动清洗(可选)，然后继续preprocess text ）