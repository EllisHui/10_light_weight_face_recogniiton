import numpy as np
import torch
from retinaface.prior_box import PriorBox, cfg_mnet, cfg_re50, PriorBox_np
from retinaface.retinaface import RetinaFace
from retinaface.box_utils import decode, decode_landm
from retinaface.py_cpu_nms import py_cpu_nms
import cv2

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(net='mnet',device='cuda'):
    if net == 'mnet':
        pretrained_path = 'retinaface/mobilenet0.25_Final.pth'
        # print('Loading pretrained model from {}'.format(pretrained_path))
        model = RetinaFace(cfg=cfg_mnet, phase='test')
    else:
        pretrained_path = 'retinaface/Resnet50_Final.pth'
        # print('Loading pretrained model from {}'.format(pretrained_path))
        model = RetinaFace(cfg=cfg_re50, phase='test')

    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    # print('Finished loading model!')
    return model

class RetinafaceDetector:
    def __init__(self, net='mnet', device='cuda'):
        self.net = net
        self.device = torch.device(device)
        self.model = load_model(net=net, device=device).to(self.device)
        self.model.eval()

    def detect_faces(self, img_raw, confidence_threshold=0.9, top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):
        img = np.float32(img_raw)
        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.model(img)  # forward pass
            # print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print(landms.shape)
        landms = landms.reshape((-1, 5, 2))
        # print(landms.shape)
        landms = landms.transpose((0, 2, 1))
        # print(landms.shape)
        landms = landms.reshape(-1, 10, )
        # print(landms.shape)

        return dets, landms

class RetinafaceDetector_dnn:
    def __init__(self, model_path='retinaface/FaceDetector_320.onnx'):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.im_height = int(model_path[:-5].split('_')[-1])
        self.im_width = int(model_path[:-5].split('_')[-1])
        priorbox = PriorBox_np(cfg_mnet, image_size=(self.im_height, self.im_width))
        self.prior_data = priorbox.forward()  ####PriorBox生成的一堆anchor在强项推理过程中始终是常数是不变量，因此只需要在构造函数里定义一次即可
        self.scale = np.array([[self.im_width, self.im_height]])
    #####使用numpy做后处理, 摆脱对pytorch的依赖
    def decode(self, loc, priors, variances):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    def decode_landm(self, pre, priors, variances):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), axis=1)
        return landms
    def detect_faces(self, img_raw, confidence_threshold=0.9, top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):
        blob = cv2.dnn.blobFromImage(img_raw, size=(self.im_width, self.im_height), mean=(104, 117, 123))
        self.model.setInput(blob)
        loc, conf, landms = self.model.forward(['loc', 'conf', 'landms'])

        boxes = self.decode(np.squeeze(loc, axis=0), self.prior_data, cfg_mnet['variance'])
        boxes = boxes * np.tile(self.scale, (1, 2)) / resize    ####广播法则
        scores = np.squeeze(conf, axis=0)[:, 1]
        landms = self.decode_landm(np.squeeze(landms, axis=0), self.prior_data, cfg_mnet['variance'])
        landms = landms * np.tile(self.scale, (1, 5)) / resize   ####广播法则

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print(landms.shape)
        landms = landms.reshape((-1, 5, 2))
        # print(landms.shape)
        landms = landms.transpose((0, 2, 1))
        # print(landms.shape)
        landms = landms.reshape(-1, 10, )
        # print(landms.shape)
        srcim_scale = np.array([[img_raw.shape[1], img_raw.shape[0]]]) / self.scale
        dets[:, :4] = dets[:, :4] * np.tile(srcim_scale, (1, 2))    ###还原到原图上
        # landms = landms * np.tile(srcim_scale, (1, 5))    ####5个关键点坐标是x1,y1,x2,y2,x3,y3,x4,y4,x5,y5排列
        landms = landms * np.repeat(srcim_scale, 5, axis=1)    ####5个关键点坐标是x1,x2,x3,x4,x5,y1,y2,y3,y4,y5排列
        return dets, landms

def convert_onnx():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    long_side = 640

    pretrained_path = 'retinaface/mobilenet0.25_Final.pth'
    model = RetinaFace(cfg=cfg_mnet, phase='test')
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()

    output_onnx = 'retinaface/FaceDetector_'+str(long_side)+'.onnx'
    inputs = torch.randn(1, 3, long_side, long_side).to(device)
    torch.onnx.export(model, inputs, output_onnx, output_names=['loc', 'conf', 'landms'])
    print('convert retinaface to onnx finish!!!')