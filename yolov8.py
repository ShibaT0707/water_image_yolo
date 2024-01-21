import os
import re
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import random
import cv2

background_image_directory = 'background_images'
image_directory = 'train_images'
image_val_directory = 'val_images'
text_directory = 'train_xml'
text_val_directory = 'val_xml'
output_directory = 'ts/images/train'
output_text_directory = 'ts/labels/train'
output_val_directory = 'ts/images/val'
output_val_text_directory = 'ts/labels/val'
if not os.path.exists(image_directory):
  os.mkdir(image_directory)
if not os.path.exists(image_val_directory):
  os.mkdir(image_val_directory)
if not os.path.exists(text_directory):
  os.mkdir(text_directory)
if not os.path.exists(text_val_directory):
  os.mkdir(text_val_directory)
if not os.path.exists(output_directory):
  os.mkdir('ts')
  os.mkdir('ts/images')
  os.mkdir(output_directory)

if not os.path.exists(output_text_directory):
  os.mkdir('ts/labels')
  os.mkdir(output_text_directory)

if not os.path.exists(output_val_directory):
  os.mkdir(output_val_directory)

if not os.path.exists(output_val_text_directory):
  os.mkdir(output_val_text_directory)

images_name = []
val_images_name = []
images_name_sorted = []
val_images_name_sorted = []
images_name_only = []
val_images_name_only = []
images_num = []
val_images_num = []
class_nums = []
class_num = 0
new_width = 640
num = 30000
img_width = [0] * num
img_height = [0] * num
img_width2 = [0] * num
img_height2 = [0] * num
img_width3 = [0] * num
img_height3 = [0] * num
img_width4 = [0] * num
img_height4 = [0] * num
img_width5 = [0] * num
img_height5 = [0] * num
img_width6 = [0] * num
img_height6 = [0] * num

while_max_num = 100
while_num = 0


name_num = 0


max_num = 0
over_num = 0


class_nums_for_unique_names = []

image_directory_first = 'images'
angle = 30
pca_num = 20
angle_num_max = 12
image_train_path = "train_imgages"
image_val_path = "val_images"
width = 1
height = 1
center_x = 0.5
center_y = 0.5

max_beta = 50
min_beta = -50


new_name_num = 0

#画像取得
def get_file_name_images():
    global images_num,images_name_only,images_name_sorted,inages_name,images_name,image,image_directory
    # ディレクトリ内のファイルをリストアップ
    files = os.listdir(image_directory_first)

    # 画像ファイルのリストをフィルタリング
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    # 画像ファイルの名前を表示
    for image_file in image_files:
        images_name.append(image_file)

    # ファイル名を数値としてソート
    sorted_image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group(0)))

    # ソートされたファイル名を表示
    for image_file in sorted_image_files:
        images_name_sorted.append(image_file)
        # 数字より前の文字列のみを抽出
        images_name_only.append(re.search(r'\D+', image_file).group(0))

        images_num.append(int(re.search(r'\d+', image_file).group(0)))



def image_change_light(image_array_input):

    assert image_array_input.ndim == 3 and (image_array_input.shape[2] == 3 or image_array_input.shape[2] == 4)
    assert image_array_input.dtype == np.uint8

    if image_array_input.shape[2] == 4:
        # RGBA画像の場合はアルファチャンネルを無視してRGBのみを取り出す
        image_array_input = image_array_input[:, :, :3]

    img = image_array_input.reshape(-1, 3).astype(np.float32)
    img = (img - np.mean(img, axis=0)) / np.std(img, axis=0)

    cov = np.cov(img, rowvar=False)
    lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)

    rand = np.random.randn(3) * 0.1
    delta = np.dot(p_eigen_vector, rand * lambd_eigen_value)
    delta = delta.reshape(1, 1, 3)

    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)


    return img_out




def rotate():
    global original_image,rotated_image,w,h
    original_image = Image.open('rotate.png')
    w, h = original_image.size

    rotated_image = Image.new('RGB',(3*w,3*h), color=(255,255,255))

    rotated_image.paste(original_image, (w, h))

    rotated_image = rotated_image.rotate(angle=30, resample=Image.BICUBIC, expand=False)

    rotated_image = rotated_image.crop((w,h,2*w,2*h))



    return np.array(rotated_image, dtype=np.uint8)

#保存
def save_image_first(image_array, num, i):
    global images_num,images_name_only,images_name_sorted,inages_name,images_name,image
    edited_image = Image.fromarray(image_array)
    if int(images_num[i]) > 7:
        edited_image.save(os.path.join(image_val_directory, f"{images_name_only[i]}{num + 1}.png"))
        #txtファイルの作成

        class_id = class_nums_for_unique_names[i]
        annotation_data = f"{class_id} {center_x} {center_y} {width} {height}"
        with open(os.path.join(text_val_directory, f"{images_name_only[i]}{num + 1}.txt"), mode='w') as f:
            f.write(annotation_data)
    else:
        edited_image.save(os.path.join(image_directory, f"{images_name_only[i]}{num + 1}.png"))
        class_id = class_nums_for_unique_names[i]
        annotation_data = f"{class_id} {center_x} {center_y} {width} {height}"
        with open(os.path.join(text_directory, f"{images_name_only[i]}{num + 1}.txt"), mode='w') as f:
            f.write(annotation_data)
#アフィン変換
def affine(target_image, ratio, distnatoin):
        global ww,hh
        # アフィン変換（上下、左右から見た傾き）
        h, w, _ = target_image.shape
        if distnatoin == 0:
            distortion_w = int(ratio * w)
            x1 = 0
            x2 = 0 - distortion_w
            x3 = w
            x4 = w - distortion_w
            y1 = h
            y2 = 0
            y3 = 0
            y4 = h
            # 変換後のイメージのサイズ
            ww = w + distortion_w
            hh = h
        elif distnatoin == 1:
            distortion_w = int(ratio * w)
            x1 = 0 - distortion_w
            x2 = 0
            x3 = w - distortion_w
            x4 = w
            y1 = h
            y2 = 0
            y3 = 0
            y4 = h
            # 変換後のイメージのサイズ
            ww = w + distortion_w
            hh = h
        elif distnatoin == 2:
            distortion_h = int(ratio * h)
            x1 = 0
            x2 = 0
            x3 = w
            x4 = w
            y1 = h
            y2 = 0 - int(distortion_h * 0.6)
            y3 = 0
            y4 = h - distortion_h
            # 変換後のイメージのサイズ
            ww = w
            hh = h + int(distortion_h * 1.3)
        elif distnatoin == 3:
            distortion_h = int(ratio * h)
            x1 = 0
            x2 = 0
            x3 = w
            x4 = w
            y1 = h - int(distortion_h * 0.6)
            y2 = 0
            y3 = 0 - distortion_h
            y4 = h
            # 変換後のイメージのサイズ
            ww = w
            hh = h + int(distortion_h * 1.3)

        pts2 = [(x2, y2), (x1, y1), (x4, y4), (x3, y3)]
        w2 = max(pts2, key=lambda x: x[0])[0]
        h2 = max(pts2, key=lambda x: x[1])[1]
        h, w, _ = target_image.shape
        pts1 = np.float32([(0, 0), (0, h), (w, h), (w, 0)])
        pts2 = np.float32(pts2)

        M = cv2.getPerspectiveTransform(pts2, pts1)
        target_image = cv2.warpPerspective(
            target_image, M, (w2 + 100, h2 + 100), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0)
        )
        #target_image = cv2.resize(target_image, (w, h))

        return (target_image, ww, hh)
def get_file_name():
    global file_names_without_extension,background_images_max_num,background_images_files,images_name,images_name_sorted,images_name_only,images_num,val_images_name,val_images_name_only
    global val_images_num,val_images_name_sorted, val_file_names_without_extension
    # ディレクトリ内のファイルをリストアップ
    files = os.listdir(image_directory)
    val_files = os.listdir(image_val_directory)
    background_files = os.listdir(background_image_directory)

    # 画像ファイルのリストをフィルタリング
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    val_image_files = [f for f in val_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    background_images_files = [f for f in background_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

    background_images_max_num = int(len(background_images_files))

    print(background_images_files)

    # 画像ファイルの名前を表示
    for image_file in image_files:
        images_name.append(image_file)
    for val_image_file in val_image_files:
        val_images_name.append(val_image_file)

    # ファイル名を数値としてソート
    sorted_image_files = sorted(image_files, key=lambda x: int(re.search(r'\d+', x).group(0)))
    sorted_val_image_files = sorted(val_image_files, key=lambda x: int(re.search(r'\d+', x).group(0)))
    file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in sorted_image_files]
    val_file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in sorted_val_image_files]


    # ソートされたファイル名を表示
    for image_file in sorted_image_files:
        images_name_sorted.append(image_file)
        # 数字より前の文字列のみを抽出
        images_name_only.append(re.search(r'\D+', image_file).group(0))

        images_num.append(int(re.search(r'\d+', image_file).group(0)))
    for val_image_file in sorted_val_image_files:
        val_images_name_sorted.append(val_image_file)
        # 数字より前の文字列のみを抽出
        val_images_name_only.append(re.search(r'\D+', val_image_file).group(0))

        val_images_num.append(int(re.search(r'\d+', val_image_file).group(0)))
#画像結合
def combine_images(i):

    global img_width, img_height, img_width2, img_height2, img_width3, img_height3, img_width4, img_height4, img_width5, img_height5, img_width6, img_height6, name_num, while_num
    global name_num,new_name_num


    images = []
    calassids = []
    count = i * 3
    data = []
    new_data1 = []
    new_data2 = []
    # 画像フォルダから画像を読み込む

    for j in range(3):
        try:
            image_converted = np.array(Image.open(os.path.join(image_directory, images_name_sorted[j + count])), dtype=np.uint8)




            img_path = os.path.join(image_directory, images_name_sorted[j + count])
            xml_path = os.path.join(text_directory, file_names_without_extension[j + count] + '.txt')



            img = Image.open(img_path)
            images.append(img)

            with open(text_directory + '/' + file_names_without_extension[j + count] + '.txt', 'r') as file:
                line = file.readline()
                class_id_str = line.split()[0]
                class_id = int(class_id_str)
                calassids.append(class_id)
        except:
            pass


    # 画像を横に3枚、縦に2枚結合
    num_horizontal = 3
    num_vertical = 2
    img_height_num = 0

    # 各画像の幅と高さを取得


    if len(images) >= 1:
        img_width[i] = images[0].width
        img_height[i] = images[0].height
    rdm_centerx_1 = random.randint(0, 640)
    rdm_centery_1 = random.randint(0, 480)

    if len(images) >= 2:
        img_width2[i] = images[1].width
        img_height2[i] = images[1].height

    rdm_centerx_2 = random.randint(0, 640)
    rdm_centery_2 = random.randint(0, 480)

    if len(images) >= 3:
        img_width3[i] = images[2].width
        img_height3[i] = images[2].height

    rdm_centerx_3 = random.randint(0, 640)
    rdm_centery_3 = random.randint(0, 480)


    if len(images) >= 4:
        img_width4[i] = images[3].width
        img_height4[i] = images[3].height

    rdm_centerx_4 = random.randint(0, 6400)
    rdm_centery_4 = random.randint(0, 480)

    if len(images) >= 5:
        img_width5[i] = images[4].width
        img_height5[i] = images[4].height

    rdm_centerx_5 = random.randint(0, 640)
    rdm_centery_5 = random.randint(0, 480)


    rdm_centerx_6 = random.randint(0, 640)
    rdm_centery_6 = random.randint(0, 480)

    if len(images) >= 6:
        img_width6[i] = images[5].width
        img_height6[i] = images[5].height


        # 結合された画像を作成
    rdm_num =  random.randint(0,background_images_max_num - 1)
    combined_image = Image.open(os.path.join(background_image_directory, background_images_files[rdm_num]))

    #写真を1920*1080にリサイズ
    combined_image = combined_image.resize((640, 480))




    combined_image_centerx = combined_image.width / 2
    combined_image_centery = combined_image.height / 2


    x = 1
    y = 1

    image1_overlap_flag = False
    image2_overlap_flag = False
    image3_overlap_flag = False

    flag2 = False
    flag3 = False

    while_num = 0



    while image2_overlap_flag == False:

        if (
        # 画像1と画像2の重なりを検出
            (rdm_centerx_2 - img_width2[i] // 2 <= rdm_centerx_1 + img_width[i] // 2 and
            rdm_centerx_2 + img_width2[i] // 2 >= rdm_centerx_1 - img_width[i] // 2 and
            rdm_centery_2 - img_height2[i] // 2 <= rdm_centery_1 + img_height[i] // 2 and
            rdm_centery_2 + img_height2[i] // 2 >= rdm_centery_1 - img_height[i] // 2) or

            # 画像1と画像3の重なりを検出
            (rdm_centerx_2 - img_width2[i] // 2 <= rdm_centerx_3 + img_width3[i] // 2 and
            rdm_centerx_2 + img_width2[i] // 2 >= rdm_centerx_3 - img_width3[i] // 2 and
            rdm_centery_2 - img_height2[i] // 2 <= rdm_centery_3 + img_height3[i] // 2 and
            rdm_centery_2 + img_height2[i] // 2 >= rdm_centery_3 - img_height3[i] // 2)


        ):
            # 重なりが検出された場合、位置を調整
                rdm_centerx_2 = random.randint(0, 640)
                rdm_centery_2 = random.randint(0, 480)
        else:
            image2_overlap_flag = True
        while_num += 1
        if while_num > while_max_num:
            image2_overlap_flag = True
            flag2 = True
            while_num = 0


    while image3_overlap_flag == False:
        if (
        # 画像1と画像2の重なりを検出
            (rdm_centerx_3 - img_width3[i] // 2 <= rdm_centerx_2 + img_width2[i] // 2 and
            rdm_centerx_3 + img_width3[i] // 2 >= rdm_centerx_2 - img_width2[i] // 2 and
            rdm_centery_3 - img_height3[i] // 2 <= rdm_centery_2 + img_height2[i] // 2 and
            rdm_centery_3 + img_height3[i] // 2 >= rdm_centery_2 - img_height2[i] // 2) or

            # 画像1と画像3の重なりを検出
            (rdm_centerx_3 - img_width3[i] // 2 <= rdm_centerx_1 + img_width[i] // 2 and
            rdm_centerx_3 + img_width3[i] // 2 >= rdm_centerx_1 - img_width[i] // 2 and
            rdm_centery_3 - img_height3[i] // 2 <= rdm_centery_1 + img_height[i] // 2 and
            rdm_centery_3 + img_height3[i] // 2 >= rdm_centery_1 - img_height[i] // 2)

        ):
            # 重なりが検出された場合、位置を調整
            rdm_centerx_3 = random.randint(0, 640)
            rdm_centery_3 = random.randint(0, 480)
        else:
            image3_overlap_flag = True
        while_num += 1
        if while_num > while_max_num:
            image3_overlap_flag = True
            flag3 = True
            while_num = 0






    img_widths = [img_width[i], img_width2[i], img_width3[i], img_width4[i], img_width5[i], img_width6[i]]
    img_heights = [img_height[i], img_height2[i], img_height3[i], img_height4[i], img_height5[i], img_height6[i]]




    # 結合された画像のサイズを計算
    total_width = 1920
    total_height = 1080


    if(flag2 == False and flag3 == False):

        if len(images) >= 1:

            combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

            img_width[i] = img_width[i] / total_width
            img_height[i] = img_height[i] / total_height

            if (img_width[i] > 1):
                img_width[i] = 1
            if (img_height[i] > 1):
                img_height[i] = 1



            img_centerx1 = rdm_centerx_1 / total_width
            img_centery1 = rdm_centery_1 / total_height

            if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                data.append(new_data1)




        if len(images) >= 2:

            combined_image.paste(images[1], (int(rdm_centerx_2 - img_width2[i] // 2 ), int(rdm_centery_2 - img_height2[i]//2)), images[1])

            img_centerx2 = rdm_centerx_2 / total_width
            img_centery2 = rdm_centery_2 / total_height
            img_width2[i] = img_width2[i] / total_width
            img_height2[i] = img_height2[i] / total_height

            if (img_width2[i] > 1):
                img_width2[i] = 1
            if (img_height2[i] > 1):
                img_height2[i] = 1

            if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]
                data.append(new_data2)

        if len(images) >= 3:

            combined_image.paste(images[2], (int(rdm_centerx_3 - img_width3[i] // 2 ), int(rdm_centery_3 - img_height3[i]//2)), images[2])

            img_centerx3 = rdm_centerx_3 / total_width
            img_centery3 = rdm_centery_3 / total_height
            img_width3[i] = img_width3[i] / total_width
            img_height3[i] = img_height3[i] / total_height

            if (img_width3[i] > 1):
                img_width3[i] = 1
            if (img_height3[i] > 1):
                img_height3[i] = 1
            if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]
                data.append(new_data3)




        # 一枚の画像として保存

        combined_image.save(output_directory + '/' + str(name_num) +'.png')
        with open(output_text_directory + '/' + str(name_num) + '.txt', 'w') as file:
            for line in data:
                class_id, x_center, y_center, width, height = line
                yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                file.write(yolo_line + "\n")
        name_num += 1

    elif(flag2 == True and flag3 == False):

        if len(images) >= 1:


            combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

            img_width[i] = img_width[i] / total_width
            img_height[i] = img_height[i] / total_height

            if (img_width[i] > 1):
                img_width[i] = 1
            if (img_height[i] > 1):
                img_height[i] = 1



            img_centerx1 = rdm_centerx_1 / total_width
            img_centery1 = rdm_centery_1 / total_height

            if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                data.append(new_data1)




        if len(images) >= 2:

            new_combined_image = combined_image

            new_combined_image.paste(images[1], (int(combined_image_centerx), int(combined_image_centery)), images[1])

            img_centerx2 = rdm_centerx_2 / total_width
            img_centery2 = rdm_centery_2 / total_height
            img_width2[i] = img_width2[i] / total_width
            img_height2[i] = img_height2[i] / total_height

            if (img_width2[i] > 1):
                img_width2[i] = 1
            if (img_height2[i] > 1):
                img_height2[i] = 1

            if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]

            new_combined_image.save(output_directory + '/' + 'new' +  str(new_name_num) +'.png')
            with open(output_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                    class_id, x_center, y_center, width, height = new_data2
                    yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                    file.write(yolo_line + "\n")
            new_name_num += 1


        if len(images) >= 3:

            combined_image.paste(images[2], (int(rdm_centerx_3 - img_width3[i] // 2 ), int(rdm_centery_3 - img_height3[i]//2)), images[2])

            img_centerx3 = rdm_centerx_3 / total_width
            img_centery3 = rdm_centery_3 / total_height
            img_width3[i] = img_width3[i] / total_width
            img_height3[i] = img_height3[i] / total_height

            if (img_width3[i] > 1):
                img_width3[i] = 1
            if (img_height3[i] > 1):
                img_height3[i] = 1
            if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]
                data.append(new_data3)

        elif(flag2 == False and flag3 == True):

            ###
            if len(images) >= 1:


                combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

                img_width[i] = img_width[i] / total_width
                img_height[i] = img_height[i] / total_height

                if (img_width[i] > 1):
                    img_width[i] = 1
                if (img_height[i] > 1):
                    img_height[i] = 1



                img_centerx1 = rdm_centerx_1 / total_width
                img_centery1 = rdm_centery_1 / total_height

                if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                    new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                    data.append(new_data1)




            if len(images) >= 2:

                combined_image.paste(images[1], (int(rdm_centerx_2 - img_width2[i] // 2 ), int(rdm_centery_2 - img_height2[i]//2)), images[1])

                img_centerx2 = rdm_centerx_2 / total_width
                img_centery2 = rdm_centery_2 / total_height
                img_width2[i] = img_width2[i] / total_width
                img_height2[i] = img_height2[i] / total_height

                if (img_width2[i] > 1):
                    img_width2[i] = 1
                if (img_height2[i] > 1):
                    img_height2[i] = 1

                if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                    new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]
                    data.append(new_data2)



            if len(images) >= 3:


                new_combined_image = combined_image


                new_combined_image.paste(images[2], (int(combined_image_centerx), int(combined_image_centery)), images[2])

                img_centerx3 = rdm_centerx_3 / total_width
                img_centery3 = rdm_centery_3 / total_height
                img_width3[i] = img_width3[i] / total_width
                img_height3[i] = img_height3[i] / total_height

                if (img_width3[i] > 1):
                    img_width3[i] = 1
                if (img_height3[i] > 1):
                    img_height3[i] = 1
                if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                    new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]


                new_combined_image.save(output_directory + '/' + 'new' +  str(new_name_num) +'.png')
                with open(output_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                        class_id, x_center, y_center, width, height = new_data3
                        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                        file.write(yolo_line + "\n")
                new_name_num += 1
                ###

            elif(flag2 == True and flag3 == True):

                if len(images) >= 1:


                    combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

                    img_width[i] = img_width[i] / total_width
                    img_height[i] = img_height[i] / total_height

                    if (img_width[i] > 1):
                        img_width[i] = 1
                    if (img_height[i] > 1):
                        img_height[i] = 1



                    img_centerx1 = rdm_centerx_1 / total_width
                    img_centery1 = rdm_centery_1 / total_height

                    if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                        new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                        data.append(new_data1)






                if len(images) >= 2:

                    new_combined_image = combined_image

                    new_combined_image.paste(images[1], (int(combined_image_centerx), int(combined_image_centery)), images[1])

                    img_centerx2 = rdm_centerx_2 / total_width
                    img_centery2 = rdm_centery_2 / total_height
                    img_width2[i] = img_width2[i] / total_width
                    img_height2[i] = img_height2[i] / total_height

                    if (img_width2[i] > 1):
                        img_width2[i] = 1
                    if (img_height2[i] > 1):
                        img_height2[i] = 1

                    if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                        new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]

                    new_combined_image.save(output_directory + '/' + 'new' +  str(new_name_num) +'.png')
                    with open(output_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                            class_id, x_center, y_center, width, height = new_data2
                            yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                            file.write(yolo_line + "\n")
                    new_name_num += 1


                if len(images) >= 3:

                    new_combined_image = combined_image


                    new_combined_image.paste(images[2], (int(combined_image_centerx), int(combined_image_centery)), images[2])

                    img_centerx3 = rdm_centerx_3 / total_width
                    img_centery3 = rdm_centery_3 / total_height
                    img_width3[i] = img_width3[i] / total_width
                    img_height3[i] = img_height3[i] / total_height

                    if (img_width3[i] > 1):
                        img_width3[i] = 1
                    if (img_height3[i] > 1):
                        img_height3[i] = 1
                    if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                        new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]


                    new_combined_image.save(output_directory + '/' + 'new' +  str(new_name_num) +'.png')
                    with open(output_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                            class_id, x_center, y_center, width, height = new_data3
                            yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                            file.write(yolo_line + "\n")
                    new_name_num += 1


        # 一枚の画像として保存

        combined_image.save(output_directory + '/' + str(name_num) +'.png')
        with open(output_text_directory + '/' + str(name_num) + '.txt', 'w') as file:
            for line in data:
                class_id, x_center, y_center, width, height = line
                yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                file.write(yolo_line + "\n")
        name_num += 1








name_num = 0
new_name_num = 0

def val_combine_images(i):

    global img_width, img_height, img_width2, img_height2, img_width3, img_height3, img_width4, img_height4, img_width5, img_height5, img_width6, img_height6, name_num, while_num
    global name_num,new_name_num


    images = []
    calassids = []
    count = i * 3
    data = []
    new_data1 = []
    new_data2 = []
    # 画像フォルダから画像を読み込む

    for j in range(3):
        try:
            image_converted = np.array(Image.open(os.path.join(image_val_directory, val_images_name_sorted[j + count])), dtype=np.uint8)




            img_path = os.path.join(image_val_directory, val_images_name_sorted[j + count])
            xml_path = os.path.join(text_val_directory, val_file_names_without_extension[j + count] + '.txt')


            img = Image.open(img_path)
            images.append(img)

            with open(text_val_directory + '/' + val_file_names_without_extension[j + count] + '.txt', 'r') as file:
                line = file.readline()
                class_id_str = line.split()[0]
                class_id = int(class_id_str)
                calassids.append(class_id)
        except:

            pass


    # 画像を横に3枚、縦に2枚結合
    num_horizontal = 3
    num_vertical = 2
    img_height_num = 0

    # 各画像の幅と高さを取得


    if len(images) >= 1:
        img_width[i] = images[0].width
        img_height[i] = images[0].height
    rdm_centerx_1 = random.randint(0, 640)
    rdm_centery_1 = random.randint(0, 480)

    if len(images) >= 2:
        img_width2[i] = images[1].width
        img_height2[i] = images[1].height

    rdm_centerx_2 = random.randint(0, 640)
    rdm_centery_2 = random.randint(0, 480)

    if len(images) >= 3:
        img_width3[i] = images[2].width
        img_height3[i] = images[2].height

    rdm_centerx_3 = random.randint(0, 640)
    rdm_centery_3 = random.randint(0, 480)


    if len(images) >= 4:
        img_width4[i] = images[3].width
        img_height4[i] = images[3].height

    rdm_centerx_4 = random.randint(0, 640)
    rdm_centery_4 = random.randint(0, 480)

    if len(images) >= 5:
        img_width5[i] = images[4].width
        img_height5[i] = images[4].height

    rdm_centerx_5 = random.randint(0, 640)
    rdm_centery_5 = random.randint(0, 480)


    rdm_centerx_6 = random.randint(0, 640)
    rdm_centery_6 = random.randint(0, 480)

    if len(images) >= 6:
        img_width6[i] = images[5].width
        img_height6[i] = images[5].height


        # 結合された画像を作成
    rdm_num =  random.randint(0,background_images_max_num - 1)
    combined_image = Image.open(os.path.join(background_image_directory, background_images_files[rdm_num]))

    #写真を1920*1080にリサイズ
    combined_image = combined_image.resize((640, 480))





    combined_image_centerx = combined_image.width / 2
    combined_image_centery = combined_image.height / 2

    x = 1
    y = 1

    image1_overlap_flag = False
    image2_overlap_flag = False
    image3_overlap_flag = False
    flag2 = False
    flag3 = False


    while_num = 0



    while image2_overlap_flag == False:


        if (
        # 画像1と画像2の重なりを検出
            (rdm_centerx_2 - img_width2[i] // 2 <= rdm_centerx_1 + img_width[i] // 2 and
            rdm_centerx_2 + img_width2[i] // 2 >= rdm_centerx_1 - img_width[i] // 2 and
            rdm_centery_2 - img_height2[i] // 2 <= rdm_centery_1 + img_height[i] // 2 and
            rdm_centery_2 + img_height2[i] // 2 >= rdm_centery_1 - img_height[i] // 2) or

            # 画像1と画像3の重なりを検出
            (rdm_centerx_2 - img_width2[i] // 2 <= rdm_centerx_3 + img_width3[i] // 2 and
            rdm_centerx_2 + img_width2[i] // 2 >= rdm_centerx_3 - img_width3[i] // 2 and
            rdm_centery_2 - img_height2[i] // 2 <= rdm_centery_3 + img_height3[i] // 2 and
            rdm_centery_2 + img_height2[i] // 2 >= rdm_centery_3 - img_height3[i] // 2)


        ):
            # 重なりが検出された場合、位置を調整
                rdm_centerx_2 = random.randint(0, 640)
                rdm_centery_2 = random.randint(0, 480)
        else:
            image2_overlap_flag = True
        while_num += 1
        if while_num > while_max_num:
            image2_overlap_flag = True
            flag2 = True

            while_num = 0


    while image3_overlap_flag == False:

        if (
        # 画像1と画像2の重なりを検出
            (rdm_centerx_3 - img_width3[i] // 2 <= rdm_centerx_2 + img_width2[i] // 2 and
            rdm_centerx_3 + img_width3[i] // 2 >= rdm_centerx_2 - img_width2[i] // 2 and
            rdm_centery_3 - img_height3[i] // 2 <= rdm_centery_2 + img_height2[i] // 2 and
            rdm_centery_3 + img_height3[i] // 2 >= rdm_centery_2 - img_height2[i] // 2) or

            # 画像1と画像3の重なりを検出
            (rdm_centerx_3 - img_width3[i] // 2 <= rdm_centerx_1 + img_width[i] // 2 and
            rdm_centerx_3 + img_width3[i] // 2 >= rdm_centerx_1 - img_width[i] // 2 and
            rdm_centery_3 - img_height3[i] // 2 <= rdm_centery_1 + img_height[i] // 2 and
            rdm_centery_3 + img_height3[i] // 2 >= rdm_centery_1 - img_height[i] // 2)

        ):
            # 重なりが検出された場合、位置を調整
            rdm_centerx_3 = random.randint(0, 640)
            rdm_centery_3 = random.randint(0, 480)
        else:
            image3_overlap_flag = True
        while_num += 1
        if while_num > while_max_num:
            image3_overlap_flag = True
            flag3 = True

            while_num = 0






    img_widths = [img_width[i], img_width2[i], img_width3[i], img_width4[i], img_width5[i], img_width6[i]]
    img_heights = [img_height[i], img_height2[i], img_height3[i], img_height4[i], img_height5[i], img_height6[i]]




    # 結合された画像のサイズを計算
    total_width = 1920
    total_height = 1080

    print(flag2)
    print(flag3)
    if(flag2 == False and flag3 == False):

        if len(images) >= 1:

            combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

            img_width[i] = img_width[i] / total_width
            img_height[i] = img_height[i] / total_height

            if (img_width[i] > 1):
                img_width[i] = 1
            if (img_height[i] > 1):
                img_height[i] = 1



            img_centerx1 = rdm_centerx_1 / total_width
            img_centery1 = rdm_centery_1 / total_height

            if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                data.append(new_data1)




        if len(images) >= 2:

            combined_image.paste(images[1], (int(rdm_centerx_2 - img_width2[i] // 2 ), int(rdm_centery_2 - img_height2[i]//2)), images[1])

            img_centerx2 = rdm_centerx_2 / total_width
            img_centery2 = rdm_centery_2 / total_height
            img_width2[i] = img_width2[i] / total_width
            img_height2[i] = img_height2[i] / total_height

            if (img_width2[i] > 1):
                img_width2[i] = 1
            if (img_height2[i] > 1):
                img_height2[i] = 1

            if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]
                data.append(new_data2)

        if len(images) >= 3:

            combined_image.paste(images[2], (int(rdm_centerx_3 - img_width3[i] // 2 ), int(rdm_centery_3 - img_height3[i]//2)), images[2])

            img_centerx3 = rdm_centerx_3 / total_width
            img_centery3 = rdm_centery_3 / total_height
            img_width3[i] = img_width3[i] / total_width
            img_height3[i] = img_height3[i] / total_height

            if (img_width3[i] > 1):
                img_width3[i] = 1
            if (img_height3[i] > 1):
                img_height3[i] = 1
            if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]
                data.append(new_data3)




        # 一枚の画像として保存

        combined_image.save(output_val_directory + '/' + str(name_num) +'.png')
        with open(output_val_text_directory + '/' + str(name_num) + '.txt', 'w') as file:
            for line in data:
                class_id, x_center, y_center, width, height = line
                yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                file.write(yolo_line + "\n")
        name_num += 1

    elif(flag2 == True and flag3 == False):

        if len(images) >= 1:


            combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

            img_width[i] = img_width[i] / total_width
            img_height[i] = img_height[i] / total_height

            if (img_width[i] > 1):
                img_width[i] = 1
            if (img_height[i] > 1):
                img_height[i] = 1



            img_centerx1 = rdm_centerx_1 / total_width
            img_centery1 = rdm_centery_1 / total_height

            if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                data.append(new_data1)




        if len(images) >= 2:

            new_combined_image = combined_image

            new_combined_image.paste(images[1], (int(combined_image_centerx), int(combined_image_centery)), images[1])

            img_centerx2 = rdm_centerx_2 / total_width
            img_centery2 = rdm_centery_2 / total_height
            img_width2[i] = img_width2[i] / total_width
            img_height2[i] = img_height2[i] / total_height

            if (img_width2[i] > 1):
                img_width2[i] = 1
            if (img_height2[i] > 1):
                img_height2[i] = 1

            if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]

            new_combined_image.save(output_val_directory + '/' + 'new' +  str(new_name_num) +'.png')
            with open(output_val_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                    class_id, x_center, y_center, width, height = new_data2
                    yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                    file.write(yolo_line + "\n")
            new_name_num += 1


        if len(images) >= 3:

            combined_image.paste(images[2], (int(rdm_centerx_3 - img_width3[i] // 2 ), int(rdm_centery_3 - img_height3[i]//2)), images[2])

            img_centerx3 = rdm_centerx_3 / total_width
            img_centery3 = rdm_centery_3 / total_height
            img_width3[i] = img_width3[i] / total_width
            img_height3[i] = img_height3[i] / total_height

            if (img_width3[i] > 1):
                img_width3[i] = 1
            if (img_height3[i] > 1):
                img_height3[i] = 1
            if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]
                data.append(new_data3)

        elif(flag2 == False and flag3 == True):

            ###
            if len(images) >= 1:


                combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

                img_width[i] = img_width[i] / total_width
                img_height[i] = img_height[i] / total_height

                if (img_width[i] > 1):
                    img_width[i] = 1
                if (img_height[i] > 1):
                    img_height[i] = 1



                img_centerx1 = rdm_centerx_1 / total_width
                img_centery1 = rdm_centery_1 / total_height

                if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                    new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                    data.append(new_data1)




            if len(images) >= 2:

                combined_image.paste(images[1], (int(rdm_centerx_2 - img_width2[i] // 2 ), int(rdm_centery_2 - img_height2[i]//2)), images[1])

                img_centerx2 = rdm_centerx_2 / total_width
                img_centery2 = rdm_centery_2 / total_height
                img_width2[i] = img_width2[i] / total_width
                img_height2[i] = img_height2[i] / total_height

                if (img_width2[i] > 1):
                    img_width2[i] = 1
                if (img_height2[i] > 1):
                    img_height2[i] = 1

                if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                    new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]
                    data.append(new_data2)



            if len(images) >= 3:

                new_combined_image = combined_image


                new_combined_image.paste(images[2], (int(combined_image_centerx), int(combined_image_centery)), images[2])

                img_centerx3 = rdm_centerx_3 / total_width
                img_centery3 = rdm_centery_3 / total_height
                img_width3[i] = img_width3[i] / total_width
                img_height3[i] = img_height3[i] / total_height

                if (img_width3[i] > 1):
                    img_width3[i] = 1
                if (img_height3[i] > 1):
                    img_height3[i] = 1
                if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                    new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]


                new_combined_image.save(output_val_directory + '/' + 'new' +  str(new_name_num) +'.png')
                with open(output_val_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                        class_id, x_center, y_center, width, height = new_data3
                        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                        file.write(yolo_line + "\n")
                new_name_num += 1
                ###

            elif(flag2 == True and flag3 == True):

                if len(images) >= 1:


                    combined_image.paste(images[0], (int(rdm_centerx_1 - img_width[i] // 2), int(rdm_centery_1 - img_height[i] // 2)), images[0])

                    img_width[i] = img_width[i] / total_width
                    img_height[i] = img_height[i] / total_height

                    if (img_width[i] > 1):
                        img_width[i] = 1
                    if (img_height[i] > 1):
                        img_height[i] = 1



                    img_centerx1 = rdm_centerx_1 / total_width
                    img_centery1 = rdm_centery_1 / total_height

                    if (img_centerx1 <= 1.0 and img_centery1 <= 1.0):
                        new_data1 = [calassids[0], img_centerx1, img_centery1, img_width[i], img_height[i]]
                        data.append(new_data1)






                if len(images) >= 2:

                    new_combined_image = combined_image

                    new_combined_image.paste(images[1], (int(combined_image_centerx), int(combined_image_centery)), images[1])

                    img_centerx2 = rdm_centerx_2 / total_width
                    img_centery2 = rdm_centery_2 / total_height
                    img_width2[i] = img_width2[i] / total_width
                    img_height2[i] = img_height2[i] / total_height

                    if (img_width2[i] > 1):
                        img_width2[i] = 1
                    if (img_height2[i] > 1):
                        img_height2[i] = 1

                    if (img_centerx2 <= 1.0 and img_centery2 <= 1.0):
                        new_data2 = [calassids[1] if len(calassids) > 1 else 0, img_centerx2, img_centery2, img_width2[i], img_height2[i]]
                    print('保存')
                    new_combined_image.save(output_val_directory + '/' + 'new' +  str(new_name_num) +'.png')
                    with open(output_val_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                            class_id, x_center, y_center, width, height = new_data2
                            yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                            file.write(yolo_line + "\n")
                    new_name_num += 1


                if len(images) >= 3:

                    new_combined_image = combined_image


                    new_combined_image.paste(images[2], (int(combined_image_centerx), int(combined_image_centery)), images[2])

                    img_centerx3 = rdm_centerx_3 / total_width
                    img_centery3 = rdm_centery_3 / total_height
                    img_width3[i] = img_width3[i] / total_width
                    img_height3[i] = img_height3[i] / total_height

                    if (img_width3[i] > 1):
                        img_width3[i] = 1
                    if (img_height3[i] > 1):
                        img_height3[i] = 1
                    if (img_centerx3 <= 1.0 and img_centery3 <= 1.0):
                        new_data3 = [calassids[2] if len(calassids) > 2 else 0, img_centerx3, img_centery3, img_width3[i], img_height3[i]]


                    new_combined_image.save(output_val_directory + '/' + 'new' +  str(new_name_num) +'.png')
                    with open(output_val_text_directory + '/' + 'new' + str(new_name_num) + '.txt', 'w') as file:
                            class_id, x_center, y_center, width, height = new_data3
                            yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                            file.write(yolo_line + "\n")
                    new_name_num += 1


        # 一枚の画像として保存

        combined_image.save(output_val_directory + '/' + str(name_num) +'.png')
        with open(output_val_text_directory + '/' + str(name_num) + '.txt', 'w') as file:
            for line in data:
                class_id, x_center, y_center, width, height = line
                yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
                file.write(yolo_line + "\n")
        name_num += 1

min_scale = 0.1
max_scale = 1.0
images_name = []
images_name_sorted = []
images_name_only = []
images_num = []
data = []
new_width = 640
get_file_name_images()
unique_class_names = list(set(images_name_only))

class_num = 0
for i in range(len(unique_class_names)):
     class_nums.append(class_num)
     new_data1 = [class_num,unique_class_names[i]]
     data.append(new_data1)
     class_num += 1

with open('ts.yaml', 'w',encoding='utf-8') as file:
    file.write("path: /content/ts\n" )
    file.write("train: images/train\n" )
    file.write("val: images/val\n" )
    file.write("nc: " + str(len(unique_class_names)) + "\n" )
    for line in data:
        class_num,clas_name= line
        yolo_line = f"{class_num}: {clas_name}"
        file.write(yolo_line + "\n")

unique_class_names = list(set(images_name_only))
class_nums_for_unique_names = [class_nums[unique_class_names.index(name)] for name in images_name_only]

for i in range(len(images_name_only)):
  try:
        print(str(i) + '/' + str(len(images_name_only)))


        image_converted = np.array(Image.open(os.path.join(image_directory_first, images_name_sorted[i])), dtype=np.uint8)


        horizital_image = Image.open(os.path.join(image_directory_first, images_name_sorted[i])).transpose(Image.FLIP_LEFT_RIGHT)
        rotate_image = Image.open(os.path.join(image_directory_first, images_name_sorted[i]))
        horizital_image.save('horizital_image.png')
        rotate_image.save('rotate.png')
        num = 1 if int(images_num[i]) == 1 else 241 * (int(images_num[i]) - 1)

        for j in range(2):
            for k in range(angle_num_max):
                for l in range(pca_num):


                    image_converted = np.array(Image.open(os.path.join("rotate.png")), dtype=np.uint8)

                     # ランダムなスケールを適用
                    scale = np.clip(random.uniform(min_scale, max_scale), 0.3, 1.0)
                    distonation = random.randint(0, 3)
                    ratio = random.uniform(0, 0.1)


                    if distonation == 2 or distonation == 3:
                       ratio = random.uniform(0, 0.3)
                    (image_converted, w, h) = affine(image_converted, ratio, distonation)

                    # 画像の幅を 640 に縮小
                    aspect_ratio = new_width / image_converted.shape[1]
                    new_height = int(image_converted.shape[0] * aspect_ratio)
                    resized_img = Image.fromarray(image_converted).resize((new_width, new_height))

                    # スケールの適用
                    s_w, s_h = resized_img.size
                    resized_img = resized_img.resize((int(s_w * scale), int(s_h * scale)))



                    image_converted = np.array(resized_img, dtype=np.uint8)


                    beta = np.random.uniform(min_beta, max_beta)

                    image_converted = Image.fromarray(image_converted)


                    image_converted.save('touka.png')

                    path = "touka.png"


                    src = cv2.imread(path)

                    # Point 1: 白色部分に対応するマスク画像を生成
                    mask = np.all(src[:,:,:] == [255, 255, 255], axis=-1)

                    # Point 2: 元画像をBGR形式からBGRA形式に変換
                    dst = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)



                    # Point3: マスク画像をもとに、白色部分を透明化
                    dst[mask,3] = 0

                    #明るさ変更
                    #dst = cv2.addWeighted(src, 1.0, np.zeros_like(src), 0.0, beta)

                    cv2.imwrite("dst.png", dst)

                    # Point 1: 透明部分に対応するマスク画像を生成
                    mask = (dst[:, :, 3] == 0)



                    # Point 3: 透明でない部分を明るさ変更
                    dst[~mask, :3] = cv2.addWeighted(dst[~mask, :3], 1.0, np.zeros_like(dst[~mask, :3]), 0.0, beta)

                    # 保存
                    cv2.imwrite("dst.png", dst)



                    image_converted = np.array(Image.open("dst.png"), dtype=np.uint8)

                    save_image_first(image_converted, num, i)
                    num += 1

                image_converted = rotate()


                image_converted = Image.fromarray(image_converted)

                image_converted.save('rotate.png')

                path = "rotate.png"


                src = cv2.imread(path)



                image_converted = np.array(Image.open(os.path.join("rotate.png")), dtype=np.uint8)



        image_converted = horizital_image


  except:
    pass

get_file_name()
unique_class_names = list(set(images_name_only))

for i in range(len(unique_class_names)):
     class_nums.append(class_num)
     class_num += 1

unique_class_names = list(set(images_name_only))
class_nums_for_unique_names = [class_nums[unique_class_names.index(name)] for name in images_name_only]




max_num = len(images_name_only)//3
over_num = len(images_name_only)%3
print(max_num)

for i in range(max_num):
    print(str(i) + '/' + str(max_num))
    combine_images(i)

max_num = len(val_images_name_only)//3
over_num = len(val_images_name_only)%3

for i in range(max_num):
    print(str(i) + '/' + str(max_num))
    val_combine_images(i)


