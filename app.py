from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os, uuid

app = Flask(__name__)

# 資料夾設定
UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 載入 YOLO 模型
model_path = os.path.join(os.getcwd(), "best.pt")
model = YOLO(model_path)

# 食物名稱中英文對照（請根據你的 data.yaml 類別補齊）
item_translation = {
    'rice': '米飯',
    'fried cabbage': '炒高麗菜',
    'scrambled eggs with tomatoes': '番茄炒蛋',
    'stir fried water spinach': '炒空心菜',
    'dongpo pork': '東坡肉',
    'pan fried salmon': '煎鮭魚',
    'pumpkin scrambled eggs': '南瓜炒蛋',
    'braised bamboo shoots': '滷筍絲',
    'stir fried enoki mushrooms': '炒金針菇',
    'stir fried rapeseed': '炒油菜',
    'stir-fried rapeseed': '炒油菜',
    'Fried sausages': '煎香腸',
    'Stir-fried bean sprouts': '炒豆芽菜',
    'Stir fried bean sprouts': '炒豆芽菜',
    'Stir-fried carrots': '炒紅蘿蔔',
    'Stir fried carrots': '炒紅蘿蔔',

}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return "未上傳圖片", 400

    file = request.files['image']
    if file.filename == '':
        return "未選擇圖片", 400

    # 儲存上傳圖片
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)
    print(f"✅ 上傳圖片儲存於: {img_path}")

    # YOLO 偵測
    results = model(img_path)
    result = results[0]

    # 繪製結果圖像
    result_array = result.plot()
    result_image = Image.fromarray(result_array)

    # 儲存結果圖片
    result_filename = f"result_{uuid.uuid4()}.jpg"
    result_img_path = os.path.join(RESULT_FOLDER, result_filename)
    result_image.save(result_img_path)
    print(f"✅ 偵測結果儲存於: {result_img_path}")

    # 網頁路徑轉換
    uploaded_image_web = img_path.replace("\\", "/")
    result_image_web = result_img_path.replace("\\", "/")

    # 取得偵測類別與信心分數
    detected_items = []
    if result.boxes:
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            raw_name = model.names[int(cls)].strip()
            normalized_name = raw_name.lower().replace("_", " ").replace("-", " ")


            # 嘗試找中譯名
            zh_name = item_translation.get(normalized_name)

            # 若找不到，試著忽略底線或空白差異
            if zh_name is None:
                for key in item_translation.keys():
                    if key.replace("_", " ").lower() == normalized_name:
                        zh_name = item_translation[key]
                        break

            # 若仍找不到，fallback 為原英文名
            if zh_name is None:
                zh_name = raw_name

            print(f"偵測到: {raw_name} -> 對應中文: {zh_name}")
            display = f"{raw_name}（{zh_name}） - 信心值：{conf:.2f}"
            detected_items.append(display)

        detected_chinese = sorted(list(set(detected_items)))
    else:
        detected_chinese = ["未偵測到食物"]

    return render_template("index.html",
                           uploaded_image=uploaded_image_web,
                           result_image=result_image_web,
                           detected_chinese=detected_chinese)

if __name__ == "__main__":
    app.run(debug=True)
