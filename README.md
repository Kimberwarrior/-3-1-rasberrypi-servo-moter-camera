# [프로젝트LAB 3-1 기말자료] 라즈베리파이를 이용한 장치조절

### 주요기술
teachable machine

gTTS

### 작품동기
여름에 에어컨의 사용비중이 높다고 판단하여 어떻게 하면 에너지소비를 최소화하면서 효율을 극대화 할 수 있을까 생각하면서 만든 장치이다.
서보모터를 에어컨으로 가정하고 서보모터가 설치된 방안에는 공기의 흐름이 방밖으로 빠져나가지 않도록 출입구를 카메라로 실시간으로 감시하여
개방되었는지 폐쇄되었는지를 시각적으로 확인, gTTs를 통해 청각적으로도 확인한다. 문이 개방된 상태라면 서보모터는 정지, 패쇄된 상태라면 서보모터는 동작한다.


### 블럭도
<p align="center">
<img src="https://user-images.githubusercontent.com/61779129/174603516-b23c90dd-4188-46e6-beeb-7e820713674f.PNG">
</p>

### 시연영상
<p align="center">
<img src="https://user-images.githubusercontent.com/61779129/174724467-1e6b5d01-ac7f-4148-b6d6-369296e2c044.jpg">
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/61779129/174593139-22e175c3-1435-4af8-8c7b-ecccdb52d200.gif">
<img src="https://user-images.githubusercontent.com/61779129/174593278-88d29684-80a0-428e-b189-33e628bb367c.gif">
</p>
https://www.youtube.com/watch?v=ZgRJuLpXYf8

### 소스코드
```c

import argparse
import sys
import time
import pygame

import cv2
from cv2 import waitKey
from image_classifier import ImageClassifier
from image_classifier import ImageClassifierOptions

from sklearn.utils import indexable
from gtts import gTTS # gtts

from pymata4 import pymata4                 # ARDUINO + SERVO
#SERVO INFORMATION
DELAY = 1
MIN = 5
MAX = 175
MID = 90
board = pymata4.Pymata4()

servo = board.set_pin_mode_servo(11) # 11번핀을 서보모터 신호선으로 설정

def move_servo(v):                  # 파이선 함수 정의
    board.servo_write(11, v)
    time.sleep(1)

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


def run(model: str, max_results: int, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:
  
  # Initialize the image classification model
  options = ImageClassifierOptions(
      num_threads=num_threads,
      max_results=max_results,
      enable_edgetpu=enable_edgetpu)
  classifier = ImageClassifier(model, options)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    # Calculate the FPS
    """
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()
    """
    # Show the FPS
    """
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (_LEFT_MARGIN, _ROW_SIZE)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    """

    counter += 1
    image = cv2.flip(image, 1)
    # List classification results
    categories = classifier.classify(image)
    # Show classification results on the image
    for idx, category in enumerate(categories):
      class_name = category.label
      score = round(category.score, 2)
      result_text = class_name + ' (' + str(score) + ')'
      text_location = (_LEFT_MARGIN, (idx + 2) * _ROW_SIZE)
      cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                  _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    print(categories[0].label)
    cv2.imshow('image_classification', image)
    # door status = closed
    if(categories[0].label=='0 door_Close'):
      f=open("door_Close.txt",'r')
      myText=f.read().replace('\n',' ')
      language="en"
      output= gTTS(text=myText, lang=language,slow=False)
      output.save("door_Status.mp3")
      pygame.mixer.init()
      pygame.mixer.music.load("door_Status.mp3")
      pygame.mixer.music.play()
      f=open("door_Close.txt",'r')
      print(f.read())
      f.close()
      cv2.waitKey(3000)
      break
    # door status = open
    if(categories[0].label=='1 door_Open'):
      f=open("door_Open.txt",'r')
      myText=f.read().replace('\n',' ')
      language="en"
      output= gTTS(text=myText, lang=language,slow=False)
      output.save("door_Status.mp3")
      pygame.mixer.init()
      pygame.mixer.music.load("door_Status.mp3")
      pygame.mixer.music.play()
      f=open("door_Open.txt",'r')
      print(f.read())
      f.close()
      cv2.waitKey(3000)
      if cv2.waitKey(1) == 27: # keep pressing the ESC key.
        break
# Stop the program if the ESC key is pressed.
  cap.release()
  cv2.destroyAllWindows()
  
  while(1):
    if(categories[0].label=='1 door_Open'):
      break
    if(categories[0].label=='0 door_Close'):
      move_servo(MAX)
      move_servo(MIN)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of image classification model.',
      required=False,
      default='model_result.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max of classification results.',
      required=False,
      default=3)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.maxResults), int(args.numThreads),
      bool(args.enableEdgeTPU), int(args.cameraId), args.frameWidth,
      args.frameHeight)


if __name__ == '__main__':
  main()

```
