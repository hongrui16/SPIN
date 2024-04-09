"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import time

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(ori_img, bbox_file = None, openpose_file = None, input_res=224, bbox = None):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    bbox: [x_min, y_min, x_max, y_max] 
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = ori_img[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        if bbox is not None:
            img = ori_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        new_size = max(width, height)
        # center = np.array([width // 2, height // 2])
        # scale = max(height, width) / 200
        # # print('height: ', height, 'width: ', width, 'center: ', center, 'scale: ', scale)
        new_img = np.ones((new_size, new_size, 3), dtype=np.uint8)
        xmin = (new_size - width) // 2
        ymin = (new_size - height) // 2
        new_img[ymin:ymin+height, xmin:xmin+width] = img

        img = cv2.resize(new_img, (input_res, input_res))
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
        img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img


def yolo_detect_person(frame, yolo_model, device, vis_results=False):
    """Use YOLO to detect a person in the frame."""
    ori_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
    frame = np.transpose(frame, (2, 0, 1))  # HWC到CHW
    frame = np.expand_dims(frame, axis=0)  # 增加批次维度
    frame = frame / 255.0  # 归一化
    frame = torch.tensor(frame).float()  # 转换为torch tensor
    frame = frame.to(device)  # 移动到GPU

    # 进行检测
    with torch.no_grad():
        results = yolo_model(frame)

    for det in results.xyxy[0]:
        if det[5] == 0:  # 在COCO数据集中，类别0通常是人
            xmin, ymin, xmax, ymax = map(int, det[:4])
            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            if vis_results:
                cv2.rectangle(ori_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    if vis_results:
        # 显示结果
        cv2.imshow('YOLOv5 Detection', frame)

        if cv2.waitKey(1) == ord('q'):  # 按q退出
            cv2.destroyAllWindows()

    return [xmin, ymin, xmax, ymax]


def cv2_ssd_detect_person(image, ssd_net, device, vis_results=False):
    """Use YOLO to detect a person in the frame."""
    h, w, _ = image.shape
    # 将图片转换为模型输入的格式
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # 输入数据并进行前向传播
    ssd_net.setInput(blob)
    detections = ssd_net.forward()



        # 遍历检测结果
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # 筛选出高置信度的检测结果
        if confidence > 0.2:
            # 获取类别ID，如果检测到的是人，则class_id为15
            class_id = int(detections[0, 0, i, 1])
            
            # 只处理人体检测结果
            if class_id == 15:
                # 计算边界框的坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                if vis_results:
                    # 绘制边界框和置信度
                    label = "{}: {:.2f}%".format("Person", confidence * 100)
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow("Output", image)

                    if cv2.waitKey(0) == ord('q'):  # 按q退出
                        cv2.destroyAllWindows()

    return [startX, startY, endX, endY]



def run_demo(hmr_model, smpl, renderer, img_filepath, device, output_dir, bbox_file=None, openpose_file=None, ssd_net=None):
    
    ori_img = cv2.imread(img_filepath) # BGR image
    if ssd_net is not None:
        start_time = time.time()
        bbox = cv2_ssd_detect_person(ori_img.copy(), ssd_net, device, vis_results=False)
        print('ssd detect time: ', time.time() - start_time)
    img, norm_img = process_image(ori_img.copy(), bbox_file, openpose_file, input_res=constants.IMG_RES, bbox=bbox)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = hmr_model(norm_img.to(device))
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        
    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()

    # print('begin rendering')
    # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img, True)
    
    # print('begin rendering side')
    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center
    
    # Render non-parametric shape
    img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))


    img_name_prefix = os.path.basename(img_filepath).split('.')[0]
    # Save reconstructions
    shape_filepath = os.path.join(output_dir, img_name_prefix + '_shape.jpg') 
    # print('shape_filepath: ', shape_filepath)
    cv2.imwrite(shape_filepath, 255 * img_shape[:,:,::-1])

    shape_side_filepath = os.path.join(output_dir, img_name_prefix + '_shape_side.jpg')
    # print('shape_side_filepath: ', shape_side_filepath)
    cv2.imwrite(shape_side_filepath, 255 * img_shape_side[:,:,::-1])

def main(args):
    
    img_filepath = args.img
    bbox_file = args.bbox
    openpose_file = args.openpose
    input_dir = args.inputdir
    output_dir = args.outdir
    
    os.makedirs(output_dir, exist_ok=True)
    # args.checkpoint = r'C:\Users\hongr\Documents\GMU_research\computerVersion\hand_modeling\VIBE\data\vibe_data\spin_model_checkpoint.pth.tar'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    hmr_model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    hmr_model.load_state_dict(checkpoint['model'], strict=False)

    # yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    # yolo_model.eval()

    ssd_net = cv2.dnn.readNetFromCaffe('ssd_caffe/MobileNetSSD_deploy.prototxt',
                               'ssd_caffe/MobileNetSSD_deploy.caffemodel')


    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    hmr_model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)


    # Preprocess input image and generate predictions

    if input_dir is not None:
        img_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        for img_file in img_files:
            print('Processing: ', img_file)
            run_demo(hmr_model, smpl, renderer, img_file, device, output_dir, bbox_file, openpose_file, ssd_net)
    elif img_filepath is not None:
        run_demo(hmr_model, smpl, renderer, img_filepath, device, output_dir, bbox_file, openpose_file, ssd_net)
    else:
        print('No input image provided. Please provide either --img or --inputdir argument.')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default= 'data/model_checkpoint.pt' , help='Path to pretrained checkpoint')
    parser.add_argument('--img', type=str, help='Path to input image')
    parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
    parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
    parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
    parser.add_argument('--outdir', type=str, default='output', help='output directory for the results')
    parser.add_argument('--inputdir', type=str, default=None, help='input directory for the images')

    args = parser.parse_args()
    main(args)
    