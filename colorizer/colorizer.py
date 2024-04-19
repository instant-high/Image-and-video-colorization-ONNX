import cv2
import onnxruntime
import numpy as np

class COLORIZER:
    def __init__(self, model_path="colorizer_siggraph17.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        #self.resolution = self.session.get_inputs()[0].shape[-2:]


    def colorize(self, img_bgr, r_factor):
        # preprocess:
        if img_bgr.ndim == 2:
            img_bgr = np.tile(img_bgr[:, :, None], 3)
        img_bgr_orig = img_bgr  # uint8
        img_bgr_rs = cv2.resize(img_bgr_orig,(r_factor,r_factor)) # w, h

        img_bgr_rs_norm = img_bgr_rs.astype(np.float32) / 255.0
        img_bgr_orig_norm = img_bgr_orig.astype(np.float32) / 255.0

        img_lab_orig = cv2.cvtColor(img_bgr_orig_norm, cv2.COLOR_BGR2Lab)
        img_lab_rs = cv2.cvtColor(img_bgr_rs_norm, cv2.COLOR_BGR2Lab)

        img_l_orig = img_lab_orig[:, :, 0]
        img_l_rs = img_lab_rs[:, :, 0]
        
        img_l_rs = np.expand_dims(img_l_rs, 0)
        img_l_rs = np.expand_dims(img_l_rs, 0) # (1,1,256,256)
        
        # inference:
        out_ab = self.session.run(['output'], input_feed={"input": img_l_rs})[0]
        
        # postprocess:
        HW_orig = img_l_orig.shape[:]  # H_orig x W_orig
        HW = out_ab.shape[2:]  # 1 x 2 x H x W
        out_ab = out_ab[0].transpose(1, 2, 0)
        
        if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
            out_ab_orig = cv2.resize(out_ab, (HW_orig[1], HW_orig[0]))
        else:
            out_ab_orig = out_ab

        out_a_orig = out_ab_orig[:, :, 0]
        out_b_orig = out_ab_orig[:, :, 1]
        out_lab_img = cv2.merge([img_l_orig, out_a_orig, out_b_orig])

        out_bgr_img_norm = cv2.cvtColor(out_lab_img, cv2.COLOR_Lab2BGR)
        out_bgr_img_norm = out_bgr_img_norm * 255.0
        out_bgr_img = out_bgr_img_norm.astype(np.uint8)
                
        return out_bgr_img