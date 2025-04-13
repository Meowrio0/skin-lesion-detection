import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------
# 設定路徑 (請根據您的實際情況修改)
# ---------------------------
# 原始資料所在的根目錄（請依照實際路徑修改）
BASE_DIR = '/Users/linjianxun/Desktop/嵌入式seg專題/Hospital Italiano Skin Lesions 2019-2022(分類模型的皮膚病變資料集）'
IMAGE_DIR = os.path.join(BASE_DIR, 'images')      # 所有圖片都存放在此資料夾內
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')   # CSV 檔案存放位置

# 建立輸出資料夾：這裡我們將在桌面上建立一個新的「split_dataset」資料夾
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
OUTPUT_DIR = os.path.join(desktop_dir, "split_dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 訓練集與測試集子資料夾
train_dir = os.path.join(OUTPUT_DIR, 'train')
test_dir = os.path.join(OUTPUT_DIR, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ---------------------------
# 讀取 CSV 與篩選資料
# ---------------------------
# 取得 metadata 資料夾內的 CSV 檔案（假設只有一個）
csv_files = [f for f in os.listdir(METADATA_DIR) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("找不到 CSV 檔案！")
csv_path = os.path.join(METADATA_DIR, csv_files[0])

# 讀取 CSV 檔案
df = pd.read_csv(csv_path)

# 刪除缺失值，並僅保留您所需要的診斷類別
df = df.dropna(subset=['isic_id', 'diagnosis'])
allowed_classes = ['nevus', 'basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']
df = df[df['diagnosis'].isin(allowed_classes)]

# 根據 isic_id 建立圖片的完整路徑（假設圖片檔名格式為 "isic_id.jpg"）
df['filename'] = df['isic_id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))

# 只保留檔案實際存在的資料
df = df[df['filename'].apply(os.path.exists)]

# ---------------------------
# 分割資料 (Stratified Train-Test Split)
# ---------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])
print("訓練集樣本數:", len(train_df))
print("測試集樣本數:", len(test_df))

# ---------------------------
# 複製圖片到對應的資料夾
# ---------------------------
def copy_images(dataframe, dest_root):
    """
    將 dataframe 中的圖片依照 diagnosis 複製到 dest_root 下各自的子資料夾中
    """
    for idx, row in dataframe.iterrows():
        class_name = row['diagnosis']
        src_path = row['filename']
        # 在目標資料夾中建立該類別的子資料夾
        dest_class_dir = os.path.join(dest_root, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        dest_path = os.path.join(dest_class_dir, os.path.basename(src_path))
        shutil.copy(src_path, dest_path)

print("正在複製訓練集圖片...")
copy_images(train_df, train_dir)
print("正在複製測試集圖片...")
copy_images(test_df, test_dir)

print("資料分割完成！")
print("訓練集圖片存放於:", train_dir)
print("測試集圖片存放於:", test_dir)