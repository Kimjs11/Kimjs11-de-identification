# Kimjs11-de-identification

# open '.ipynb' file
<b/>


번호판, 얼굴을 모자이크하는 코드입니다.  
<b/>
process: check image list - plate mosaic - face mosaic  
2-stage 
1) 번호판 Yolov5 모델 검출(자체학습) -> mosaic <b/>
2) 얼굴 DSFD모델 (pretrained-weights) -> mosaic  <b/>
각 검출 코드안에 mosaic 코드를 내장시켜 간소화하였습니다. <b/>
input image folder: './images/input' <b/>
output image folder: './images/output' <b/>
