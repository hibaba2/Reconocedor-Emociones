import cv2
from deepface import DeepFace
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer el cuadro de la cámara
    ret, frame = cap.read()

    # Realizar el reconocimiento facial en el cuadro
    try:
        results = DeepFace.analyze(frame, actions=['emotion'])

        # Extraer las emociones del primer resultado
        emotions = results[0]['emotion']

        # Mostrar las emociones en el cuadro
        dicOriginal=['angry','disgust','fear','happy','sad','surprise','neutral']
        dicEspanol=['Enojo    ',
                    'Disgusto ',
                    'Miedo    ',
                    'Felicidad',
                    'Tristeza ',
                    'Sorpresa ',
                    'Neutro   ']
        dd=dict(zip(dicOriginal,dicEspanol))
        cv2.putText(frame,"UTEM", (600, 30 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for idx, (emotion, score) in enumerate(emotions.items()):
            if score>30:
                cv2.putText(frame, "%4s:%1.2f"%(dd[emotion],score), (10, 30 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "%4s:%1.2f"%(dd[emotion],score), (10, 30 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            #cv2.putText(frame, f"{emotion}: {score}", (10, 30 + idx * 20),
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostrar el cuadro con las emociones
        cv2.imshow('Facial Emotion Recognition', frame)
    except:
        print('No detecta rostro')

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
