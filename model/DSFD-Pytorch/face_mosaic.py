import glob
import os
import cv2
import time
import face_detection
import argparse
import imageio

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


def run(source='./data'  # file/dir/URL/glob
        ):
    # if __name__ == "__main__":
    impath = str(source)
    # impaths = "/root/KJS/dataset/wider_face/WIDER_test/images/0--Parade"
    impath = glob.glob(os.path.join(impath, "*.jpg"))
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=3840
    )
    for img in impath:
        if img.endswith("out.jpg"): continue
        im = cv2.imread(img)
        imgs = imageio.imread(img)
        print("Processing:", img)
        t = time.time()
        dets = detector.detect(
            im[:, :, ::-1]
        )[:, :4]
        print(f"Detection time: {time.time()- t:.3f}")
        # draw_faces(im, dets) #rectangle
        crop_face=[]
        for bbox in dets:
            x0, y0, x1, y1 = [int(_) for _ in bbox]
            face_img = imgs[y0:y1,x0:x1]
            face_img = cv2.resize(face_img,(5,5))
            face_img = cv2.resize(face_img,(x1-x0,y1-y0), interpolation=cv2.INTER_CUBIC)
            imgs[y0:y1,x0:x1] = face_img
            crop_face.append(imgs)

        b, g, r = cv2.split(imgs)
        img_output = cv2.merge([r, g, b])

        imname = os.path.basename(img).split(".")[0]

        output_path = os.path.join(
            source,
            f"{imname}.jpg")

        cv2.imwrite(output_path, img_output)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./', help='file_dir')
    opt = parser.parse_args()
    # opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)