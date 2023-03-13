WEIGHTS="/share/userfile/zhuwenchen/deepsort/model/best.pt"
SOURCE="/share/userfile/zhuwenchen/deepsort/data/test01.mp4"
OUTPUT="/share/userfile/zhuwenchen/deepsort/output/output01.mp4"

python main.py --weights ${WEIGHTS} --source ${SOURCE} --output ${OUTPUT}