# -*- coding:utf-8 -*-


import io
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import requests

def cv2_imread_buffer(buffer):
    buffer = io.BytesIO(buffer)
    arr = np.frombuffer(buffer.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def preprocess_red_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    result = np.zeros_like(img)
    result[mask > 0] = img[mask > 0]
    return result


# 切割图标
def split_image_tag(img, tag_pos):
    x, y = tag_pos
    img_ = img[0:35, y - 37:y]
    return img_


# 旋转图片
def rotate_image(template, angle):
    center = (template.shape[1] // 2, template.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(template, rotation_matrix, (template.shape[1], template.shape[0]),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image


# 模板匹配
def template_match(template, img):
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val, max_loc





def show_cv2(image, title):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 创建图形窗口
    plt.figure()
    plt.imshow(image)
    # 获取图像尺寸
    height, width, _ = image.shape
    # 设置坐标轴范围
    plt.xlim(0, width)
    plt.ylim(height, 0)  # 反转 y 轴以正确显示图像方向
    plt.title(title)
    plt.axis('on')  # 显示坐标轴
    plt.show()


# 获取图标位置
def get_tag_position(bg, fp):
    match_tag_list = []

    img_1 = cv2_imread_buffer(bg)

    img_2 = cv2_imread_buffer(fp)

    img_1 = preprocess_red_image(img_1)

    img_2 = preprocess_red_image(img_2)

    def process_tag(tag_pos):

        new_template = split_image_tag(img_2, tag_pos)
        new_size = 75
        new_template = cv2.resize(new_template, (new_size, new_size))

        ocr_infos = []
        angel_size = 6

        for angle in range(-180, 180, angel_size):
            template_ = rotate_image(new_template, angle)
            max_val, max_loc = template_match(template_, img_1)
            ocr_infos.append([angle, max_val, max_loc])

        max_info = max(ocr_infos, key=lambda x: x[1])

        return max_info

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_tag, [(37, 37), (37, 74), (37, 111), (37, 148)]))

    for max_info in results:
        match_tag_list.append(list(max_info[-1]))

    return match_tag_list


# 主函数
def ocr_abs(bg_img, fp_img):
    pos_info = get_tag_position(bg_img, fp_img)
    pos_info = [[int(i[0] / 2), int(i[1] / 2)] for i in pos_info]
    return pos_info


if __name__ == '__main__':
    bg_img = requests.get(
        "https://castatic.fengkongcloud.cn/crb/icon_select/icon_select-1.0.0-set-000001/v1/4dd9bcd4aef415ee5c0f7aa6733417c0_bg.jpg").content
    fp_img = requests.get(
        "https://castatic.fengkongcloud.cn/crb/icon_select/icon_select-1.0.0-set-000001/v1/4dd9bcd4aef415ee5c0f7aa6733417c0_fg.png").content
    start_time = time.time()
    print(ocr_abs(bg_img, fp_img))
    print(f"识别耗时：{time.time() - start_time}")
