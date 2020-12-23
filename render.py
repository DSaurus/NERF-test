import taichi_three as t3
import taichi as ti
import numpy as np
import math
import json
import cv2

ti.init(ti.cpu)
obj = t3.readobj('/media/data1/shaoruizhi/zy/dataset/multi_eval4/obj_all/DATA4-1.obj')
# obj = t3.readobj('/media/data1/shaoruizhi/Multiview_Pair/metric/additional_multi_human/inference_eval_DATA3-2_3_0.obj')
color = obj['vi'][:, 3:]
obj['vi'] = obj['vi'][:, :3]
print(np.max(obj['vi'], axis=0), np.min(obj['vi'], axis=0))
print(color)
scene = t3.Scene()
model = t3.Model(obj=obj, col_n=color.shape[0])

light = t3.Light([0, 0, 1])
scene.add_light(light)
scene.add_model(model)

camera = t3.Camera(res=(512, 512), fx=850, fy=850)
scene.add_camera(camera)
scene.init()
model.type[None] = 1
model.vc.from_numpy(color)

train_json = {}

train_json["focal"] = 850
train_json["frames"] = []
for i in range(0, 360, 12):
    r = i / 180 * math.acos(-1)
    camera.set(pos=[4*math.cos(r), 0.8, 4*math.sin(r) + 0.35], target=[0, 0.8,  0.35])
    scene.render()

    ti.imwrite(camera.img, 'render/%03d.png' % i)
    ti.imwrite(camera.mask, 'render/%03d_mask.png' % i)
    img = cv2.imread('render/%03d.png' % i)
    mask = cv2.imread('render/%03d_mask.png' % i)
    img2 = np.zeros((512, 512, 4))
    img2[:, :, :3] = img
    img2[:, :, 3] = mask[:, :,  0]
    cv2.imwrite('render/%03d.png' % i, img2)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = camera.trans_py
    extrinsic[:3, 3] = camera.pos_py

    frame = {}
    frame["file_path"] = './render/%03d.png' % i
    frame["transform_matrix"] =  [[extrinsic[i][j] for j in range(4)] for i in range(4)]
    train_json["frames"].append(frame)

json.dump(train_json, open('transforms_train.json', 'w'))

test_json = {}

test_json["focal"] = 850
test_json["frames"] = []
for i in range(6, 360, 4):
    r = i / 180 * math.acos(-1)
    camera.set(pos=[4*math.cos(r), 2.0, 4*math.sin(r) + 0.35], target=[0, 0.8,  0.35])
    scene.render()

    ti.imwrite(camera.img, 'render/%03d.png' % i)
    ti.imwrite(camera.mask, 'render/%03d_mask.png' % i)
    img = cv2.imread('render/%03d.png' % i)
    mask = cv2.imread('render/%03d_mask.png' % i)
    img2 = np.zeros((512, 512, 4))
    img2[:, :, :3] = img
    img2[:, :, 3] = mask[:, :, 0]
    cv2.imwrite('render/%03d.png' % i, img2)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = camera.trans_py
    extrinsic[:3, 3] = camera.pos_py

    frame = {}
    frame["file_path"] = './render/%03d.png' % i
    frame["transform_matrix"] = [[extrinsic[i][j] for j in range(4)] for i in range(4)]
    test_json["frames"].append(frame)
json.dump(test_json, open('transforms_test.json', 'w'))
