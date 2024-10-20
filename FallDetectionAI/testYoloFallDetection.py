from ultralytics import YOLO

#This file will not work because there is no data on git, no minio integration yet because there are alot of pictures and we don't know if we can upload them to minio

def test_model():
    model = YOLO('yolov11_fall_detection.pt')

    results = model.val(data='data.yaml', imgsz=640)

    print(results)



if __name__ == '__main__':
    test_model()
