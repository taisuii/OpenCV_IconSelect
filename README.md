# OpenCV_IconSelect
shumei captcha icon select | 数美验证码点选识别破解 | 非模型纯算法识别
### 安卓逆向，JS逆向，图像识别，在线接单，全套源码+部署+算法联系QQ: 27788854，wechat: taisuivip，[telegram: rtais00](https://t.me/rtais00)
### 微信公众号：R逆向
# 0x0 前言
github开源地址：[https://github.com/taisuii/OpenCV_IconSelect](https://github.com/taisuii/OpenCV_IconSelect)
#### 验证码分析
验证码例子为
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4579b61e5e84897a68a1c674d2d73fb.png)

我们不难可以发现这几个特征点

 - 图标大小均匀没有拉伸和畸变，只进行了简单的旋转
 - 图标颜色单一且均为红色
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d0d427acc2d142f58f23eab8760a776c.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/88613406699f41fa920a2babf728abdd.png)

##### 解决方案
对于这种图像，我们可以直接使用纯算法识别，思路如下：
提取背景图红色像素部分，把小图标按X轴均匀切割，逐个匹配，或四个线程并发匹配
# 0x1 识别算法部分
##### 字节流转换为cv2图片
对于网络下载的图片进行转换以便于后续处理
```python
def cv2_imread_buffer(buffer):
    buffer = io.BytesIO(buffer)
    arr = np.frombuffer(buffer.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
```
##### 背景图红色部分提取
```python
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
```
提取后效果如下，到这一步，几乎是无脑识别了，剩下的代码就是
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/90293321d802428fafa945a1124bfce6.png)
##### 识别，匹配出坐标
切割小图标，并把小图标缩放成和背景图上大小差不多的图标
然后旋转360度，每6度匹配一次大图
```python
def split_image_tag(img, tag_pos):
    x, y = tag_pos
    img_ = img[0:35, y - 37:y]
    return img_
# 多线程识别
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
```
识别结果 300+ms，速度非常不错，使用了线程识别
	[[132, 71], [181, 20], [88, 29], [221, 97]]
	识别耗时：0.3492300510406494
# 0x3 识别测试
这里仍然采用官网去测试了100次识别，平均速度344ms，成功率84%
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8bd1d1e62a9e427da065f58a07b5b2fe.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ed6e3800c3a74745b9e22c43a71336d7.png)
