# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Autonomous Drone Detection and Tracking System based on YOLOv5

This system performs:
- Object detection using YOLOv5
- Person tracking and following
- Autonomous drone control (takeoff, search, track, land)
- LiDAR distance measurement
- PID control for smooth movement

Usage:
    $ python detect.py --mode flight --control PID --debug_path debug/run1
    $ python detect.py --mode test --source 0  # test mode with webcam
"""

import argparse
import csv
import os
import platform
import sys
import time
import collections
import numpy as np
from pathlib import Path

import torch

# Jetson Nano detection
def is_jetson_nano():
    """Detect if running on Jetson Nano"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return 'jetson nano' in model.lower()
    except:
        return False

JETSON_NANO = is_jetson_nano()
if JETSON_NANO:
    print("🚀 Jetson Nano detected - enabling optimizations")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

# Drone control modules (mock implementations for safety)
try:
    import keyboard
except ImportError:
    print("Warning: keyboard module not found. Install with: pip install keyboard")
    keyboard = None

# Mock modules for drone control (replace with actual implementations)
class MockLidar:
    def connect_lidar(self, port):
        print(f"Mock: Connected to LiDAR on {port}")
    
    def read_lidar_distance(self):
        return [2.0]  # Mock distance in meters

class MockControl:
    def __init__(self):
        self.x_delta = 0
        self.z_delta = 0
        
    def connect_drone(self, port):
        print(f"Mock: Connected to drone on {port}")
    
    def set_flight_altitude(self, altitude):
        print(f"Mock: Set flight altitude to {altitude}m")
    
    def configure_PID(self, control_type):
        print(f"Mock: Configured {control_type} controller")
    
    def initialize_debug_logs(self, path):
        print(f"Mock: Initialized debug logs at {path}")
    
    def arm_and_takeoff(self, altitude):
        print(f"Mock: Armed and taking off to {altitude}m")
    
    def land(self):
        print("Mock: Landing drone")
    
    def stop_drone(self):
        print("Mock: Stopping drone")
    
    def control_drone(self):
        print("Mock: Controlling drone movement")
    
    def set_system_state(self, state):
        print(f"Mock: Set system state to {state}")
    
    def print_drone_report(self):
        print("Mock: Drone status report")
    
    def setXdelta(self, x_delta):
        self.x_delta = x_delta
    
    def setZDelta(self, z_delta):
        self.z_delta = z_delta
    
    def getMovementYawAngle(self):
        return self.x_delta * 0.1  # Mock PID output
    
    def getMovementVelocityXCommand(self):
        return self.z_delta * 0.5  # Mock PID output

class MockVision:
    @staticmethod
    def get_single_axis_delta(center, target):
        return target - center
    
    @staticmethod
    def point_in_rectangle(point, left, right, top, bottom):
        return left <= point[0] <= right and top <= point[1] <= bottom

# Initialize mock modules
lidar = MockLidar()
control = MockControl()
vision = MockVision()

# Drone configuration
MAX_FOLLOW_DIST = 2.0  # meters
MAX_ALT = 2.5  # meters
MAX_MA_X_LEN = 5
MAX_MA_Z_LEN = 5
MA_X = collections.deque(maxlen=MAX_MA_X_LEN)  # Moving Average X
MA_Z = collections.deque(maxlen=MAX_MA_Z_LEN)  # Moving Average Z
STATE = "takeoff"  # takeoff, land, track, search


class Detection:
    """Detection class to store bounding box information"""
    def __init__(self, xyxy, conf, cls, names):
        self.Left = int(xyxy[0])
        self.Top = int(xyxy[1])
        self.Right = int(xyxy[2])
        self.Bottom = int(xyxy[3])
        self.Center = ((self.Left + self.Right) / 2, (self.Top + self.Bottom) / 2)
        self.confidence = float(conf)
        self.class_id = int(cls)
        self.class_name = names[self.class_id]

class DroneDetector:
    """YOLOv5-based detector for drone autonomous flight"""
    def __init__(self, weights=ROOT / "yolov5s.pt", device="", conf_thres=0.25, iou_thres=0.45, jetson_nano=False):
        self.device = select_device(device)
        self.jetson_nano = jetson_nano
        
        # Jetson Nano optimizations
        if jetson_nano:
            # Use smaller image size for better performance
            self.imgsz = check_img_size((416, 416), s=32)  # Smaller input size
            # Enable FP16 for better performance on Jetson
            fp16 = True
            print("🚀 Jetson Nano optimizations enabled")
        else:
            self.imgsz = check_img_size((640, 640), s=32)
            fp16 = False
            
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=ROOT / "data/coco128.yaml", fp16=fp16)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: Could not open camera, using mock data")
            self.cap = None
        
        # Warmup
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        print("YOLOv5 detector initialized")
    
    def get_detections(self):
        """Get detections from camera feed"""
        if self.cap is None:
            # Mock detection for testing
            mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(mock_image, "MOCK CAMERA", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return [], 30.0, mock_image
        
        ret, frame = self.cap.read()
        if not ret:
            return [], 0, None
        
        start_time = time.time()
        
        # Preprocess
        im = cv2.resize(frame, self.imgsz)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        # Inference
        pred = self.model(im, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0], max_det=1000)
        
        # Process detections
        detections = []
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                
                # Create Detection objects for persons only (class 0)
                for *xyxy, conf, cls in det:
                    if int(cls) == 0:  # person class
                        detection = Detection(xyxy, conf, cls, self.names)
                        detections.append(detection)
        
        fps = 1.0 / (time.time() - start_time)
        return detections, fps, frame
    
    def get_image_size(self):
        """Get camera image size"""
        if self.cap is None:
            return 640, 480
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
    
    def close_camera(self):
        """Close camera connection"""
        if self.cap:
            self.cap.release()

# Global detector instance
detector = None

def initialize_detector():
    """Initialize the YOLOv5 detector"""
    global detector
    detector = DroneDetector()

def setup():
    """Setup all drone systems"""
    print("connecting lidar")
    lidar.connect_lidar("/dev/ttyTHS1")
    
    print("setting up detector")
    initialize_detector()
    
    print("connecting to drone")
    if args.mode == "flight":
        print("MODE = flight")
        control.connect_drone('/dev/ttyACM0')
    else:
        print("MODE = test")
        control.connect_drone('127.0.0.1:14551')
    
    control.set_flight_altitude(MAX_ALT)

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source="0",  # webcam by default for drone
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    # Drone-specific parameters
    mode="test",  # flight or test mode
    control_type="PID",  # PID or P controller
    debug_path="debug/run1",  # debug output path
):
    """
    Autonomous Drone Detection and Tracking System
    
    This function runs the complete drone autonomous flight system including:
    - YOLOv5-based person detection
    - Drone state management (takeoff, search, track, land)
    - PID control for smooth tracking
    - LiDAR distance measurement
    - Real-time visualization
    """
    global STATE, detector
    
    # Initialize detector with custom parameters and Jetson Nano optimization
    detector = DroneDetector(weights=weights, device=device, conf_thres=conf_thres, iou_thres=iou_thres, jetson_nano=JETSON_NANO)
    
    # Setup drone systems
    setup_drone_systems(mode, control_type, debug_path)
    
    # Get image dimensions for visualization
    image_width, image_height = detector.get_image_size()
    image_center = (image_width / 2, image_height / 2)
    
    # Setup debug video writer
    debug_image_writer = None
    if mode == "flight":
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        debug_image_writer = cv2.VideoWriter(
            debug_path + ".avi",
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
            25.0,
            (image_width, image_height)
        )
    
    print(f"Starting autonomous drone system in {mode} mode...")
    
    # Main drone control loop
    try:
        while True:
            if STATE == "track":
                STATE = track_person(image_center, debug_image_writer, mode)
            elif STATE == "search":
                STATE = search_person(mode)
            elif STATE == "takeoff":
                STATE = takeoff_drone()
            elif STATE == "land":
                land_drone()
                break
                
    except KeyboardInterrupt:
        print("Manual interruption detected")
        land_drone()
    finally:
        if debug_image_writer:
            debug_image_writer.release()
        cv2.destroyAllWindows()

def setup_drone_systems(mode, control_type, debug_path):
    """Setup all drone subsystems"""
    print("Connecting LiDAR...")
    lidar.connect_lidar("/dev/ttyTHS1")
    
    print("Connecting to drone...")
    if mode == "flight":
        print("MODE = flight")
        control.connect_drone('/dev/ttyACM0')
    else:
        print("MODE = test")
        control.connect_drone('127.0.0.1:14551')
    
    control.set_flight_altitude(MAX_ALT)
    control.configure_PID(control_type)
    control.initialize_debug_logs(debug_path)

def track_person(image_center, debug_image_writer, mode):
    """Track detected person and control drone movement"""
    print("State: TRACKING")
    
    while True:
        # Check for manual interruption
        if keyboard and keyboard.is_pressed('q'):
            print("Manual interruption - landing")
            return "land"
        
        # Get detections from YOLOv5
        detections, fps, image = detector.get_detections()
        
        if len(detections) > 0:
            # Track the first detected person
            person_to_track = detections[0]
            print(f"Tracking person: confidence={person_to_track.confidence:.2f}")
            
            person_center = person_to_track.Center
            
            # Calculate deltas for control
            x_delta = vision.get_single_axis_delta(image_center[0], person_center[0])
            y_delta = vision.get_single_axis_delta(image_center[1], person_center[1])
            
            # Check if LiDAR is pointing at target
            lidar_on_target = vision.point_in_rectangle(
                image_center,
                person_to_track.Left, person_to_track.Right,
                person_to_track.Top, person_to_track.Bottom
            )
            
            # Get LiDAR distance
            lidar_dist = lidar.read_lidar_distance()[0]
            
            # Update moving averages
            MA_Z.append(lidar_dist)
            MA_X.append(x_delta)
            
            # Calculate control commands
            velocity_z_command = 0
            if lidar_dist > 0 and lidar_on_target and len(MA_Z) > 0:
                z_delta_MA = calculate_moving_average(MA_Z)
                z_delta_MA = z_delta_MA - MAX_FOLLOW_DIST
                control.setZDelta(z_delta_MA)
                velocity_z_command = control.getMovementVelocityXCommand()
            
            # Yaw control
            yaw_command = 0
            if len(MA_X) > 0:
                x_delta_MA = calculate_moving_average(MA_X)
                control.setXdelta(x_delta_MA)
                yaw_command = control.getMovementYawAngle()
            
            # Execute drone control
            control.control_drone()
            
            # Visualize tracking
            prepare_visualization(
                lidar_dist, person_center, person_to_track, image,
                yaw_command, x_delta, y_delta, fps, velocity_z_command,
                lidar_on_target, image_center
            )
            
            visualize_output(image, debug_image_writer, mode)
            
        else:
            print("Lost target - switching to search")
            return "search"

def search_person(mode):
    """Search for person when target is lost"""
    print("State: SEARCHING")
    start_time = time.time()
    control.stop_drone()
    
    while time.time() - start_time < 40:  # Search for 40 seconds
        if keyboard and keyboard.is_pressed('q'):
            print("Manual interruption - landing")
            return "land"
        
        detections, fps, image = detector.get_detections()
        print(f"Searching... Found {len(detections)} detections")
        
        if len(detections) > 0:
            print("Target found - switching to track")
            return "track"
        
        # Show search status
        if mode == "test":
            time_left = 40 - (time.time() - start_time)
            cv2.putText(
                image, f"Searching target. Time left: {time_left:.1f}s",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA
            )
            visualize_output(image, None, mode)
    
    print("Search timeout - landing")
    return "land"

def takeoff_drone():
    """Execute drone takeoff sequence"""
    print("State: TAKEOFF")
    control.print_drone_report()
    control.arm_and_takeoff(MAX_ALT)
    return "search"

def land_drone():
    """Execute drone landing sequence"""
    print("State: LANDING")
    control.land()
    detector.close_camera()
    sys.exit(0)

def calculate_moving_average(ma_array):
    """Calculate moving average of array"""
    if len(ma_array) == 0:
        return 0
    return sum(ma_array) / len(ma_array)

def prepare_visualization(lidar_distance, person_center, person_to_track, image,
                         yaw_command, x_delta, y_delta, fps, velocity_x_command,
                         lidar_on_target, image_center):
    """Prepare visualization overlays on image"""
    image_height, image_width = image.shape[:2]
    
    # Draw LiDAR distance visualization
    lidar_vis_x = image_width - 50
    lidar_vis_y = image_height - 50
    lidar_vis_y2 = int(image_height - lidar_distance * 200)
    cv2.line(image, (lidar_vis_x, lidar_vis_y), (lidar_vis_x, lidar_vis_y2), 
             (0, 255, 0), thickness=10)
    
    # Draw distance text
    cv2.putText(image, f"Distance: {lidar_distance:.2f}m", 
                (image_width - 300, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Draw tracking line
    cv2.line(image, 
             (int(image_center[0]), int(image_center[1])),
             (int(person_center[0]), int(person_center[1])),
             (255, 0, 0), thickness=10)
    
    # Draw bounding box
    cv2.rectangle(image,
                  (person_to_track.Left, person_to_track.Bottom),
                  (person_to_track.Right, person_to_track.Top),
                  (0, 0, 255), thickness=10)
    
    # Draw center points
    cv2.circle(image, (int(image_center[0]), int(image_center[1])), 
               20, (0, 255, 0), thickness=-1)
    cv2.circle(image, (int(person_center[0]), int(person_center[1])), 
               20, (0, 0, 255), thickness=-1)
    
    # Draw stats
    cv2.putText(image, 
                f"FPS: {fps:.1f} Yaw: {yaw_command:.2f} Forward: {velocity_x_command:.2f}",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.putText(image, f"LiDAR on target: {lidar_on_target}",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.putText(image, f"X delta: {x_delta:.2f} Y delta: {y_delta:.2f}",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

def visualize_output(image, debug_image_writer, mode):
    """Display or save visualization output"""
    if mode == "flight" and debug_image_writer:
        debug_image_writer.write(image)
    else:
        cv2.imshow("Drone Tracking", image)
        cv2.waitKey(1)


def parse_opt():
    """Parse command-line arguments for Autonomous Drone Detection and Tracking System.

    Args:
        Standard YOLOv5 arguments plus drone-specific parameters:
        --mode (str): Flight mode - 'flight' for real drone, 'test' for simulation
        --control (str): Controller type - 'PID' or 'P'
        --debug_path (str): Path for debug output files

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Examples:
        ```python
        # Test mode with webcam
        python detect.py --mode test --source 0
        
        # Flight mode with PID controller
        python detect.py --mode flight --control PID --debug_path debug/run1
        ```
    """
    parser = argparse.ArgumentParser(description='Autonomous Drone Detection and Tracking System')
    
    # YOLOv5 detection parameters
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-format", type=int, default=0, help="save format: 0=YOLO, 1=Pascal-VOC")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    
    # Drone-specific parameters
    parser.add_argument("--mode", type=str, default="test", choices=["flight", "test"], 
                       help="Flight mode: 'flight' for real drone, 'test' for simulation")
    parser.add_argument("--control", type=str, default="PID", choices=["PID", "P"], 
                       help="Controller type: 'PID' or 'P'")
    parser.add_argument("--debug_path", type=str, default="debug/run1", 
                       help="Path for debug output files")
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes Autonomous Drone Detection and Tracking System based on provided command-line arguments.

    Args:
        opt (argparse.Namespace): Command-line arguments for drone system. See function `parse_opt` for details.

    Returns:
        None

    Notes:
        This function performs essential pre-execution checks and initiates the autonomous drone system based on user-specified
        options. The system includes YOLOv5 detection, drone control, and real-time tracking capabilities.

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    print("=" * 60)
    print("🚁 Autonomous Drone Detection and Tracking System")
    print("=" * 60)
    print(f"Mode: {opt.mode}")
    print(f"Controller: {opt.control}")
    print(f"Debug Path: {opt.debug_path}")
    print(f"Weights: {opt.weights}")
    print(f"Device: {opt.device if opt.device else 'auto'}")
    print("=" * 60)
    
    # Check requirements (excluding some packages that might not be needed)
    try:
        check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    except Exception as e:
        print(f"Warning: Could not check all requirements: {e}")
    
    # Store global args for drone system
    global args
    args = opt
    
    # Run the autonomous drone system with filtered parameters
    drone_params = {
        'weights': opt.weights,
        'source': opt.source,
        'data': opt.data,
        'imgsz': opt.imgsz,
        'conf_thres': opt.conf_thres,
        'iou_thres': opt.iou_thres,
        'max_det': opt.max_det,
        'device': opt.device,
        'view_img': opt.view_img,
        'save_txt': opt.save_txt,
        'save_format': opt.save_format,
        'save_csv': opt.save_csv,
        'save_conf': opt.save_conf,
        'save_crop': opt.save_crop,
        'nosave': opt.nosave,
        'classes': opt.classes,
        'agnostic_nms': opt.agnostic_nms,
        'augment': opt.augment,
        'visualize': opt.visualize,
        'update': opt.update,
        'project': opt.project,
        'name': opt.name,
        'exist_ok': opt.exist_ok,
        'line_thickness': opt.line_thickness,
        'hide_labels': opt.hide_labels,
        'hide_conf': opt.hide_conf,
        'half': opt.half,
        'dnn': opt.dnn,
        'vid_stride': opt.vid_stride,
        'mode': opt.mode,
        'control_type': opt.control,
        'debug_path': opt.debug_path
    }
    
    run(**drone_params)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
