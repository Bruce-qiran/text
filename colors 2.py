import cv2
import numpy as np

# 加载图像
image = cv2.imread(r'C:\Users\20601\AppData\Roaming\Tencent\QQ\Temp\997431a461970a0e8c9d455a03ac82e2.jpg')
output = image.copy()

# 转换为 HSV 颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义颜色范围 (红、青、蓝)
colors = {
    "red": ([(0, 50, 50), (10, 255, 255)],  # 红色低阈值
            [(170, 50, 50), (180, 255, 255)]),  # 红色高阈值
    "cyan": [(80, 130, 50), (100, 255, 255)],  # 青色范围
    "blue": [(100, 150, 50), (140, 255, 255)]  # 蓝色范围
}

# 遍历颜色字典进行检测
for color_name, bounds in colors.items():
    if color_name == "red":
        # 红色需要合并低阈值和高阈值
        lower1, upper1 = bounds[0]
        lower2, upper2 = bounds[1]
        mask1 = cv2.inRange(hsv, np.array(lower1, dtype="uint8"), np.array(upper1, dtype="uint8"))
        mask2 = cv2.inRange(hsv, np.array(lower2, dtype="uint8"), np.array(upper2, dtype="uint8"))
        mask = cv2.bitwise_or(mask1, mask2)  # 合并两个掩膜
    else:
        # 其他颜色只有一个范围
        lower, upper = bounds
        mask = cv2.inRange(hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
    
    # 形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # 忽略小区域
            # 绘制轮廓
            cv2.drawContours(output, [contour], -1, (0, 0, 0), 4)

            # 计算轮廓的中心
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 添加颜色名称
                cv2.putText(output, color_name, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)

# 显示结果
resize_factor = 0.5
output_image_resized = cv2.resize(output, (0, 0), fx=resize_factor, fy=resize_factor)
cv2.imshow('Result', output_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
