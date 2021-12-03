## CatSnn 사용 방법

# python version : 3.6.8
# needed library : tensorflow(version=2.6.2), numpy, matplotlib, sklearn, cv2, tqdm, os


# 1. import CatSnn
from module.cat_indivisual_snn_test import CatSnn

# 2. 고양이 이름들 : 임의 지정.
names = ["nabi", "nana", "lulu", "hi", "cati", "flower", "bella", "simba", "luna", "roky"]

# 3. 모델에서 학습된 고양이 번호 : 이번 테스트에서는 고정. 전달한 snn_cl 모델에서 학습한 번호로 추가로 학습하면 변경됨.
selections = [6, 7, 15, 18, 19, 29, 55, 57, 82, 152]

# 4. 학습에 사용된 이미지 사이즈 : 이번 테스트에서는 150으로 고정.
img_size = 150

# 5. 가중치 저장된 주소 : cat_indivisual_model_cl_weights 주소로 설정한다.
weigh_path = r"C:\Users\ADD\OneDrive - dgu.ac.kr\DGU\Codes\FarmProject\models\cat_indivisual\cat_indivisual_model_cl_weights_211202_055623.h5"

# 6. CatSnn 객체 생성
catSnn = CatSnn(num_classes=10, img_size = img_size, weighs_path = weigh_path)

# 7. 테스트할 이미지 주소 설정 : 임의로 이미지 주소를 설정한다.
img_path = r"C:\Users\ADD\OneDrive - dgu.ac.kr\DGU\Codes\FarmProject\cat_indivisual\test_img\0018_000.JPG"
# img_path = r"C:\Users\ADD\OneDrive - dgu.ac.kr\DGU\Codes\FarmProject\FarmReferenceCodes\dbscan\dbscan_faces\ind_images\0003_002.JPG"

# 8. archer 이미지 주소 설정 : 아쳐 이미지가 있는 폴더의 주소를 설정한다.
# archer 이미지 저장 형식
# archer_img
# ├── 0006
# │   └── 0006_001.JPG
# ├── 0007
# │   └── 0007_007.JPG
# :
# └── 0152
#     └── 0152_008.JPG
archer_dir=r"C:\Users\ADD\OneDrive - dgu.ac.kr\DGU\Codes\FarmProject\cat_indivisual\archer_img\\"

# 9. 결과 예측
# 9-1. 가장 유사한 고양이의 Id 받기 : predictId 함수 사용.
#      최소값이 dist_threshold 보다 커서 유사하지 않으면 unknown으로 판정된다.
#      출력 값이 -1이면 unknown 으로 판정된 것이다.
res = catSnn.predictId(selection=selections, image_path=img_path, archer_dir=archer_dir,  dist_threshold=0.4)

# 9-2. 학습된 모든 고양이와의 거리 받기 : predict 함수 사용. 값이 작을수록 유사도가 높음.
# resLst = catSnn.predict(selection=selections, image_path=img_path, dist_threshold=0.4)

# 10. 테스트 결과 출력

# 고양이 이름에서 idx가 res인 이름 출력
predicted_name=None
if res>-1:
    predicted_name=names[res]
else:
    predicted_name="unknown"
print("test img predicted name : %s" %predicted_name)

# 테스트 이미지 출력 대기
catSnn.plotImage(img_path)

# anchor 이미지 출력 대기
catSnn.plotAnchor(names)

# 출력 대기된 모든 이미지 출력
catSnn.plotShow()