import os

# 遍历指定目录下的所有文件和目录
for root, dirs, files in os.walk("/mnt/user/chenmuyin/diffsinger/github/DiffSinger-1/data/raw/xiaobing_24k/wav"):
    for filename in files:
        if "_24k" in filename:
            # 将含有_24k的文件名中的_24k去掉
            new_filename = filename.replace("_24k", "")
            # 重命名文件
            os.rename(os.path.join(root, filename), os.path.join(root, new_filename))
