import cv2
import numpy as np
from deepface import DeepFace

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Cargar la imagen del banner
image_path = 'src/banner.png'  # Reemplaza con la ruta correcta de la imagen
image = cv2.imread(image_path)

# Obtener las dimensiones originales del banner
banner_height, banner_width = image.shape[:2]

# Diccionario de traducción de emociones
dicOriginal = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
dicEspanol = ['Enojo >:(   ', 'Disgusto :S ', 'Miedo D:   ', 'Felicidad :D ', 'Tristeza :( ', 'Sorpresa :o', 'Neutro :|  ']
dd = dict(zip(dicOriginal, dicEspanol))

while True:
    # Leer el cuadro de la cámara
    ret, frame = cap.read()

    # Obtener las dimensiones del cuadro de video
    frame_height, frame_width = frame.shape[:2]

    # Ajustar la imagen a las dimensiones originales del banner
    if image is not None:
        image_resized = cv2.resize(image, (frame_width, banner_height))

        # Combinar el banner con el cuadro de video
        combined_frame = cv2.vconcat([image_resized, frame])
    else:
        combined_frame = frame

    try:
        # Realizar el reconocimiento facial en el cuadro
        results = DeepFace.analyze(frame, actions=['emotion'])

        # Extraer las emociones del primer resultado
        emotions = results[0]['emotion']

        # Obtener la emoción con el mayor valor
        max_emotion = max(emotions, key=emotions.get)

        # Crear el texto de la emoción
        emotion_text = dd[max_emotion]

        # Obtener el tamaño del texto
        text_size, _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        # Calcular la posición del texto centrado en la parte inferior
        text_x = int((frame_width - text_size[0]) / 2)
        text_y = banner_height + frame_height + 30

        # Crear una imagen en blanco con el tamaño adecuado para mostrar la emoción
        emotion_image = np.zeros((banner_height, frame_width, 3), dtype=np.uint8)
        cv2.putText(emotion_image, emotion_text, (text_x, int(banner_height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Color azul (BGR)

        # Combinar el cuadro de video con el banner y la emoción
        combined_frame = cv2.vconcat([combined_frame, emotion_image])

        # Mostrar el cuadro combinado con la emoción predominante
        cv2.imshow('Facial Emotion Recognition', combined_frame)
    except:
        print('No detecta rostro')

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
