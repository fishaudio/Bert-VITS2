
import os

def batch_rename(source,output, name, start_id):
    # 获取文件夹内的文件列表
    # files = sorted()
    files=sorted(os.listdir(source),key=lambda x:int(x.split('_')[-1].split('.')[0]))
    for idx, filename in enumerate(files, start=start_id):
        # 构建新的文件名
        new_name = f"{name}_{idx}.wav"  # 假设文件是.txt格式，您可以根据需要修改
        # 获取文件的完整路径
        old_path = os.path.join(source, filename)
        new_path = os.path.join(output, new_name)
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "--source", default="./raw/temp")
    parser.add_argument( "--output", default="./raw/兔兔")
    args=parser.parse_args()
    source=args.source
    output=args.output
    import re
    
    if (ns:=os.listdir(output)):
        baseName=ns[0].split('.')[0].split('_')[0]
        index=sorted(list(map(lambda x : int(x.split('.')[0].split('_')[1]),ns)),reverse=True)[0]
        print(f"目标路径已存在文件，查询到baseName为{baseName},index为{index}")
    else:
        print("目标路径为空，请手动输入baseName和index")
        baseName = input("Enter the base name for files: ")
        index = int(input("Enter the starting id: "))
        pass

    batch_rename(source,output, baseName, index+1)
