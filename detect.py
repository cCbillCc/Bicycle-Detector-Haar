import cv2

# --- 設定 ---
cascade_path = r'classifier\cascade.xml'
image_path = 'test_image.jpg'

# 新增設定：將圖片寬度縮小到這個尺寸，以加快偵測速度
TARGET_WIDTH = 800 

# --- 主程式 ---

# 1. 載入 Haar Cascade 分類器
bicycle_cascade = cv2.CascadeClassifier()
if not bicycle_cascade.load(cv2.samples.findFile(cascade_path)):
    print(f"錯誤：無法載入模型檔案 {cascade_path}")
    exit()

# 2. 讀取測試圖片
frame = cv2.imread(image_path)
if frame is None:
    print(f"錯誤：無法讀取圖片檔案 {image_path}")
    exit()

# ==========================================================
# === 新增的程式碼：自動縮小圖片 ===
# ==========================================================
original_height, original_width = frame.shape[:2]
# 如果圖片寬度大於我們設定的目標寬度，就進行縮小
if original_width > TARGET_WIDTH:
    # 計算縮放比例
    ratio = TARGET_WIDTH / float(original_width)
    # 計算新的高度
    target_height = int(original_height * ratio)
    # 執行縮小
    frame = cv2.resize(frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)
    print(f"圖片尺寸已從 {original_width}x{original_height} 縮小至 {TARGET_WIDTH}x{target_height}")
# ==========================================================

# 3. 將圖片轉換為灰階
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 4. 執行偵測
print("正在執行偵測，請稍候...")
bicycles = bicycle_cascade.detectMultiScale(frame_gray, scaleFactor=1.001, minNeighbors=140, minSize=(30, 30))
print("偵測完成！")

print(f"找到了 {len(bicycles)} 個可能是腳踏車的物件！")

# 5. 在原圖上畫出偵測到的矩形框
for (x, y, w, h) in bicycles:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 6. 顯示結果圖片
cv2.imshow('Bicycle Detector - Result', frame)
cv2.imwrite("result.png", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()