from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.predict("https://ultralytics.com/images/bus.jpg")

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        nome   = result.names[cls_id]
        conf   = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        print(f"{nome} — {conf:.0%} — {[round(c) for c in coords]}")



#caso vc seja um entusiasta, curioso ou estudante (que nem eu) talvez vc tenha se deparado com algum erro
#de biblioteca. Os comentários abaixo podem lhe ajudar de alguma forma:
#pip install ultralytics.
#rode esse comando no terminal caso vc ainda não tenha a versão correta da biblioteca instalada na sua IDE.
#Isso evita erros como o "ModeuleNotFoundError: no module named 'ultralytics'".
#Após instalar basta rodar o script normalmente.
