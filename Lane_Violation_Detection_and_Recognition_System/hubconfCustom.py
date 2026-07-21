import cv2
import cvzone
import torch
import numpy as np
import pandas as pd
import easyocr
import os
from datetime import datetime
from tracker import *
import time
from PIL import Image
import threading
from ultralytics import YOLO
import time
# from send_mail import prepare_and_send_email
from smtp_gmail import send_email_with_smtp_gmail


# Global Variables
is_email_allowed = False  # When user checks the email checkbox, this variable will be set to True
send_next_email = True  # We have to wait for 10 minutes before sending another email

# detections_summary will be used to store the detections summary report
detections_summary = ''

email_sender = '57224_yogeshtiwari@gbpuat-tech.ac.in'
email_recipient = 'uic.17bca1185@gmail.com'


# Mail using Google Auth Tokens

# def violation_alert_generator(im0, message, subject='Lane Violation Detected'):
#     '''This function will send an email with attached alert image and then wait for 10 minutes before sending another email
    
#     Parameters:
#       im0 (numpy.ndarray): The image to be attached in the email
#       subject (str): The subject of the email
#       message (str): The message text of the email

#     Returns:
#       None
#     '''
#     global send_next_email, email_recipient

#     message_text = message
#     send_next_email = False  # Set flag to False so that another email is not sent
#     print('Sending email alert to ', email_recipient)
#     prepare_and_send_email(email_sender, email_recipient, subject, message_text, im0)
#     print('Mail sent')
#     # Wait for 10 minutes before sending another email
#     time.sleep(600)
#     send_next_email = True


# Mail using Googe App password (SMTP)

def violation_alert_generator(im0, message, subject='Lane Violation Detected'):

    global send_next_email, email_recipient

    #create a directory to store the images that are attached to the email
    base_loc = 'static/violations/'
    location = 'GRIL Office'
    
    #get current date and time
    current_date_time = time.time()
    formatted_date_time = time.strftime("%H-%M-%S_%d-%m-%Y", time.localtime(current_date_time))
    
    #if base_loc doesn't exist, create it
    if not os.path.exists(base_loc):
        os.makedirs(base_loc)
    
    file_name = base_loc+'violation_'+location + '_' + formatted_date_time + '.jpg'

    #convert img_file into jpeg format and save it in the file_name
    cv2.imencode('.jpg', im0)[1].tofile(file_name)

    send_next_email = False

    print('Sending email alert to ', email_recipient)

    send_email_with_smtp_gmail(
                    recipient=email_recipient,
                    subject=subject,
                    body = message,
                    image_path=file_name
                )
    
    print("Email Sent")
    # Wait for 10 minutes before sending another email
    time.sleep(600)
    send_next_email = True


def increase_dpi(image, dpi=300):
    """Increase DPI of the image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = pil_image.size
    new_size = (width * dpi // 72, height * dpi // 72)
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    pil_image.save("frames/dpi_cropped_plate.png", dpi=(dpi, dpi))


def log_number_plate(ocr_results, number_plate_log):
    """
    Logs the OCR-detected number plate to a text file with a timestamp.

    Args:
        ocr_results (list): Output from EasyOCR (list of tuples)
        number_plate_log (str): Path to the output log file
    """
    plate_text = " | ".join([text[1] for text in ocr_results]) if ocr_results else "UNKNOWN"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] Number Plate: {plate_text}\n"

    with open(number_plate_log, "a") as f:
        f.write(log_line)

    return plate_text


def video_detection(conf_, frames_buffer=[]):
    print("Starting Video Detection ########################")
    '''This function will detect vehicles, display their numbers and speed in the video file or a live stream.

    Parameters:
        conf_ (float): Confidence threshold for inference
        frames_buffer (list): A list of frames to be processed
        vid_path (str): Path to the video file

    Returns:
        None
    '''
    global send_next_email
    global is_email_allowed
    global email_recipient
    global detections_summary

    # Load COCO class names
    with open("configs/coco.txt", "r") as f:
        class_list = f.read().splitlines()

    # Setup file logs
    number_plate_log = "number_plate.txt"

    # Constants for perspective transformation
    frame_width = 1600
    frame_height = 900

    # Tracker and zones
    tracker = Tracker()
    area1 = [(998, 643), (444, 643), (726, 533), (1015, 533)]
    area2 = [(977, 816), (1, 816), (228, 725), (995, 725)]

    # State
    wup = {}
    wrongway = []
    anpr_processed = []
    vehicle_number = []

    # OCR
    # ocr_weights = os.path.join(os.getcwd(), "weights", "easyocr")
    # print(ocr_weights)
    # os.environ["EASYOCR_CACHE_DIR"] = ocr_weights

    gpu_available = torch.cuda.is_available()
    reader = easyocr.Reader(['en'], model_storage_directory="weights/easyocr", gpu=gpu_available, verbose=True)

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Load models to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    print("Loading Models ########################")

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # print(BASE_DIR)
    # WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
    # print(WEIGHTS_DIR)

    # breakpoint()

    try:
        model = YOLO('weights/detection.pt').to(device)
        anpr_model = YOLO('weights/anpr_car.pt').to(device)
    except Exception as e:
        print(f"Error loading models: {e}")

    try:
        while True:
            if len(frames_buffer) > 0:
                img0 = frames_buffer.pop(0)
                if img0 is None:
                    continue
            else:
                time.sleep(0.01)
                continue

            frame = cv2.resize(img0, (frame_width, frame_height))
            results = model.predict(frame, verbose=False)
            det_data = results[0].boxes.data.cpu().numpy()
            px = pd.DataFrame(det_data).astype("float")

            detections = []
            for _, row in px.iterrows():
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                cls_id = int(row[5])
                confidence = float(row[4])
                cls_name = class_list[cls_id]

                if 'car' in cls_name and confidence > conf_:
                    detections.append([x1, y1, x2, y2])

            bbox_idx = tracker.update(detections)

            for bbox in bbox_idx:
                x3, y3, x4, y4, obj_id = bbox
                cx, cy = x4, y4

                result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
                if result >= 0:
                    wup[obj_id] = (cx, cy)

                if obj_id in wup:
                    result2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)
                    if result2 >= 0:
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)

                        if obj_id not in anpr_processed:
                            anpr_processed.append(obj_id)
                            
                            # Save vehicle crop
                            cropped_vehicle = frame[y3:y4, x3:x4]
                            save_path = "frames/vehicle.jpg"
                            cv2.imwrite(save_path, cropped_vehicle)

                            # ANPR
                            anpr_result = anpr_model.predict(save_path)
                            bbox_data = anpr_result[0].boxes.data.cpu().numpy()
                            anpr_bbox = pd.DataFrame(bbox_data).astype("float")

                            for _, row in anpr_bbox.iterrows():
                                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                                conf = float(row[4])
                                if conf > 0.5:
                                    plate_img = cv2.imread(save_path)[y1:y2, x1:x2]
                                    plate_path = "frames/cropped_plate.jpg"
                                    cv2.imwrite(plate_path, plate_img)

                                    image = cv2.imread(plate_path)
                                    increase_dpi(image)
                                    ocr_results = reader.readtext("frames/dpi_cropped_plate.png")

                                    if ocr_results not in vehicle_number:
                                        vehicle_number.append(ocr_results)

                                        plate_text = log_number_plate(ocr_results, number_plate_log)

                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        detections_summary += f"[{timestamp}] Number Plate: {plate_text}\n"

                                        message = f"Wrong Side Vehicle Detected with number plate {plate_text}"
                                        print(message)

                                        if is_email_allowed and send_next_email:    
                                            threading.Thread(target=violation_alert_generator, args=(img0, message)).start()


                        if obj_id not in wrongway:
                            wrongway.append(obj_id)

            w = len(wrongway)
            
            # Draw zones and stats
            cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 255, 255), 2)
            cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 255, 255), 2)
            cvzone.putTextRect(frame, f'Cars in Wrong Lane: {w}', (30, 100), 2, 2)

            # Yield processed frame
            yield frame, w

    except Exception as e:
        print(f"Error in video_detection: {e}")