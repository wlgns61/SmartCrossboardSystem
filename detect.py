import time
import argparse
import numpy as np

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from sig import Signal

cross_upper = [[185, 210], [530, 200], [790, 330]]
cross_lower = [[185, 210], [300, 335], [340, 337], [380, 339], [420, 341], [460, 343], [500, 345],
              [540, 347], [600, 352], [660, 343], [790, 330]]
wait_upper = [[160, 340], [300, 340], [340, 342], [380, 344], [420, 346], [460, 348], [500, 350],
              [540, 352], [580, 354], [620, 356], [660, 358], [700, 362], [740, 366], [780, 370],
              [820, 374], [854, 380]]
wait_lower = [[160, 340], [200, 480], [854, 480], [854, 380]]

cross_cat = cross_upper + list(reversed(cross_lower[1:-1]))
wait_cat = wait_upper + list(reversed(wait_lower[1:-1]))

location = (30, 50)

thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8
dsize = (608,608)

signal_cv = cv2
red = signal_cv.imread("red.jpg", cv2.IMREAD_UNCHANGED)
red = signal_cv.resize(red, dsize=dsize, interpolation=cv2.INTER_AREA)

def visualize(signal):
    if signal.signal == 1:  # green
        img = signal_cv.imread("green.jpg", cv2.IMREAD_UNCHANGED)
        img = signal_cv.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)

        if signal.extension_count < signal.max_extension:
            text = "Remain time: {}  Extendable Remaining: {}".format(signal.remain_time,
                                                                      signal.max_extension - signal.extension_count)
        else:
            text = "Remain time: {}    !No more extension!".format(signal.remain_time)
        signal_cv.putText(img, text, location, font, fontScale, (0, 0, 0), thickness)

    else:
        img = red

    signal_name = "Signal"
    signal_cv.namedWindow(signal_name)
    signal_cv.moveWindow(signal_name, 40, 30)
    signal_cv.imshow(signal_name, img)
    # signal_cv.waitKey(800)

def detect(save_img=False):

    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt, view_anot = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt, opt.view_anot
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, view_anot=view_anot, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = [[21, 243, 16] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    end = 0

    for cnt, path, img, im0s, vid_cap in dataset:
        if cnt == 1:
            print("[Signal Initialized]")
            signal = Signal(wait=5, people_exist=10, walk_time=20, extension=5)
            cv2.destroyAllWindows()

        start = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            foot_coords = []
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4], det2 = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                det[:, :4] = det[:, :4].round()
                det2[:, :4] = det2[:, :4].round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for (*xyxy, conf, cls), (*xywh) in zip(det, det2):
                    foot_coord = [xywh[0][0], (xywh[0][1]+xywh[0][3]*0.45)]
                    foot_coords.append(foot_coord)
                    cv2.line(im0, tuple(foot_coord), tuple(foot_coord), (0, 0, 255), 5)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


            # 1초에 한번씩
            if abs(start - end) >= 1.:
                signal(foot_coords, wait_upper, wait_lower,cross_upper, cross_lower)
                visualize(signal)
                end = time.time()


            # Stream results
            if view_img:
                if signal.get_signal() == 1:
                    cv2.polylines(im0, [np.array(cross_cat)], True, (255, 0, 0), 3)
                    text = "Remain time: {}".format(signal.remain_time)
                    cv2.putText(im0, text, location, font, fontScale, (255, 255, 255), thickness)
                else:
                    cv2.polylines(im0, [np.array(wait_cat)], True, (0, 0, 255), 3)

                    if signal.wait_count >= signal.wait:
                        text = "Waiting time: {}".format(signal.count)
                        cv2.putText(im0, text, location, font, fontScale, (255, 255, 255), thickness)
                    else:
                        text = "Driving time: {}".format(signal.wait_count)
                        cv2.putText(im0, text, location, font, fontScale, (255, 255, 255), thickness)

                # cv2.namedWindow(p)
                cv2.moveWindow(p, 675, 30)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--view-anot', action='store_true', help='print anotation to terminal')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
