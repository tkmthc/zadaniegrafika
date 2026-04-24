import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

local_filename = "zdjecie.jpg" 
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/800px-Lemon.jpg"

print("Próba wczytania obrazu")
img = cv2.imread(local_filename)

if img is None:
    print(f"Nie znaleziono pliku {local_filename}. Pobieram z internetu...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        print("Błąd pobierania")
        exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("1. Obraz oryginalny")
plt.axis('off')

height, width = img.shape[:2]
new_dim = (int(width * 0.5), int(height * 0.5))
resized_img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

rotated_img = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)

plt.subplot(1, 2, 2)
plt.imshow(rotated_img, cmap='gray')
plt.title("2. Wynik (-50%, Gray, Obrót)")
plt.axis('off')
plt.show()

print("\n--- MACIERZ OBRAZU (fragment 10x10) ---")
print(rotated_img[:10, :10]) 

print(f"\nPełny wymiar macierzy wynikowej: {rotated_img.shape}")
print("Gotowe")