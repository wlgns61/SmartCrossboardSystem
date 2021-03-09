# Introduction

This directory contains PyTorch YOLOv3 software developed by Ultralytics LLC, and **is freely available for redistribution under the GPL-3.0 license**. For more information please visit https://www.ultralytics.com.

# Requirements

Python 3.7 or later with all `pip install -U -r requirements.txt` packages including `torch >= 1.5`. Docker images come with all dependencies preinstalled. Docker requirements are: 
- Nvidia Driver >= 440.44
- Docker Engine - CE >= 19.03
- numpy  
- opencv-python >= 4.1  
- torch >= 1.5  
- matplotlib  
- pycocotools  
- tqdm  
- pillow  
- tensorboard >= 1.14  

# 프로젝트 개요

1. 카메라로 사람을 인지하여 신호등을 제어하는 시스템. 카메라는 현재 보행자의 신호가 빨간불일 때 신호대기 영역을 탐지하고, 파란불일 때 횡단보도 영역을 탐지함.    
2. 보행가능시간(보행자 신호가 파란불인 경우), 보행자의 신호대기 시간(보행자 신호가 빨간불인 경우), 최소 차량 통행시간(파란불 무한 유지 방지), 신호 연장 시간(보행자 통행 시간 연장), 최대 신호 연장 횟수에 대한 Hyper Parameter를 가지도록 설계  


main.py로 실행  
