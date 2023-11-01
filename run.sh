# python rename.py --source ./raw/temp --output ./raw/兔兔

# python autoLable.py
echo 1
# python autoLable.py -p ./raw/bakiMix -n bakiMix -tp ./filelists/bakiMix/raw_bakiMix.list
#python clean_list.py 但是貌似也没啥用
#python clean_raw.py -p ./filelists/tutu/raw_tutu.list -fp ./filelists/tutu/tutu.list
echo -----------------得到转录文件--------------------
# python resample.py
# python preprocess_text.py   -tp  ./filelists/c1/c1.list -cp ./configs/config_c1.json


echo -----------------完成数据预处理\(.clean,训练集和数据集的划分\)，是否继续\(1是继续训练,2是启动推理\)-----------------
read go
if [ $go -eq 1 ]
then
    echo -----------------生成bert文件--------------------
    # python bert_gen.py -c ./configs/config_c1.json 
    echo -----------------开始训练-----------------
    python train_ms.py -m c1 -c ./configs/config_c1.json 
    echo -----------------训练结束-----------------
fi
if [ $go -eq 2 ]
then
    echo -----------------开始推理-----------------
    # python webui.py -m "./logs/c1" -c "./configs/config.json"  
    streamlit run stream_run.py #配置都写到文件里了 
fi
echo -----------------结束，退出程序-----------------


#   python train_ms.py -m c1test -c ./configs/config_c1test.json 