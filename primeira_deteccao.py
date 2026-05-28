from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.5,
    save=True
)
for r in results:
    print(r.boxes)

#esse código serve para realizar a verificação em uma imagem utilizando a biblioteca YOLO.

#caso vc seja um entusiasta, curioso ou estudante (que nem eu) talvez vc tenha se deparado com algum erro
#de biblioteca. Os comentários abaixo podem lhe ajudar de alguma forma:
#pip install ultralytics.
#rode esse comando no terminal caso vc ainda não tenha a versão correta da biblioteca instalada na sua IDE.
#Isso evita erros como o "ModeuleNotFoundError: no module named 'ultralytics'".
#Após instalar basta rodar o script normalmente.
