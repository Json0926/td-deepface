from deepface import DeepFace
import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)  # 从摄像头读取视频流

while True:
    ret, frame = video_capture.read()  # 读取一帧

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测人脸
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30))

    # 对每个检测到的人脸进行识别
    for (x, y, w, h) in faces:
        # 裁剪人脸区域
        face = frame[y:y + h, x:x + w]
        print(face)

        # 使用DeepFace进行人脸识别
        # result = DeepFace.verify(face, model_name='VGG-Face', distance_metric='euclidean_l2')
        result = DeepFace.find(img_path=face,
                               db_path="./workspace/my_db",
                               model_name='Facenet',
                               distance_metric='euclidean_l2',
                               enforce_detection=False)

        # 在图像上绘制人脸边界框和识别结果
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        closest_face_path = result[0]['identity'].values[0]
        person_name = os.path.basename(os.path.dirname(closest_face_path))
        cv2.putText(frame, f'Name: {person_name}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 显示图像
    cv2.imshow('Video', frame)

    # 按下q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流和窗口
video_capture.release()
cv2.destroyAllWindows()

# models = [
#     "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"
# ]
# img1 = "./tests/dataset/img1.jpg"
# img2 = "./tests/dataset/img3.jpg"
# result = DeepFace.verify(img1, img2, model_name=models[1])
# print("Is verified: ", result["verified"])
# print(result)
# dfs = DeepFace.find(img_path=img1, db_path="./workspace/my_db")

# result = DeepFace.verify(img1, img1, model_name=models[1])
# dfs = DeepFace.find(img_path=img1,
#                     db_path="./workspace/my_db",
#                     model_name=models[1])
# # identity = dfs["identity"]
# # embedding = DeepFace.represent(img_path=img1, model_name=models[1])

# closest_face_path = dfs[0]['identity'].values[0]
# person_name = os.path.basename(os.path.dirname(closest_face_path))
# print("Person name: ", person_name)
