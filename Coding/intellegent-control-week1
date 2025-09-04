import cv2
import numpy as np

#Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Rentang warna merah dalam HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    #Masking untuk mendeteksi warna merah
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    result_red = cv2.bitwise_and(frame, frame, mask=mask_red)

    #Rentang warna biru dalam HSV
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    #Masking untuk mendeteksi warna biru
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)

    #Rentang warna hijau dalam HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    #Masking untuk mendeteksi warna hijau
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    result_green = cv2.bitwise_and(frame, frame, mask=mask_green)
    
    #Menggabungkan semua masking untuk mendeteksi warna merah, biru dan hijau
    mask = mask_red + mask_blue + mask_green
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    #Mencari kontur untuk bounding box warna merah
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_red:
        if cv2.contourArea(contour) > 500: #Mengabaikan kontur kecil
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Merah", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 2) #Memberi nama bounding box

    #Mencari kontur untuk bounding box warna biru
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_blue:
        if cv2.contourArea(contour) > 500: #Mengabaikan kontur kecil
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Biru", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 2) #Memberi nama bounding box

    #Mencari kontur untuk bounding box warna hijau
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_green:
        if cv2.contourArea(contour) > 500: #Mengabaikan kontur kecil
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Hijau", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 2) #Memberi nama bounding box

    #Menampilkan hasil
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()