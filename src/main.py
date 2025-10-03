import cv2
import numpy as np
import mediapipe as mp
import os
import argparse
import pandas as pd  # CSV oluşturmak için eklendi

def detect_eye_reflections(image):
    """Görüntüdeki göz yansımalarını tespit edip bir maske oluşturur."""
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("Yüz tespit edilemedi!")
            return mask
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Göz landmarkları
        eye_landmarks = {
            "Sol": {
                'center': 468,
                'iris': [468, 469, 470, 471, 472]  # Tüm iris noktaları
            },
            "Sağ": {
                'center': 473,
                'iris': [473, 474, 475, 476, 477]
            }
        }
        
        for eye_type in ["Sol", "Sağ"]:
            try:
                landmarks = eye_landmarks[eye_type]
                
                # İris noktalarını al
                iris_points = np.array([
                    [int(face_landmarks.landmark[idx].x * w),
                     int(face_landmarks.landmark[idx].y * h)]
                    for idx in landmarks['iris']
                ])
                
                # Göz bebeği merkezi
                pupil_center = np.array([
                    int(face_landmarks.landmark[landmarks['center']].x * w),
                    int(face_landmarks.landmark[landmarks['center']].y * h)
                ])
                
                # İris yarıçapını hesapla
                distances = [np.linalg.norm(point - pupil_center) for point in iris_points]
                iris_radius = int(max(distances))
                pupil_radius = int(iris_radius * 0.5)  # Göz bebeği yarıçapı
                
                # ROI'yi hesapla
                margin = pupil_radius
                x = int(pupil_center[0] - pupil_radius - margin)
                y = int(pupil_center[1] - pupil_radius - margin)
                size = int(2 * (pupil_radius + margin))
                
                # Sınırları kontrol et
                x = max(0, x)
                y = max(0, y)
                size = min(min(w - x, h - y), size)
                
                if size <= 0:
                    continue
                
                # ROI'yi al
                eye_roi = image[y:y+size, x:x+size]
                if eye_roi.size == 0:
                    continue
                
                # ROI'yi büyüt
                scale = 8
                eye_roi_large = cv2.resize(eye_roi, (size * scale, size * scale))
                
                # LAB renk uzayına çevir
                lab = cv2.cvtColor(eye_roi_large, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
                
                # Kontrast artır
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(l_channel)
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=20)
                
                # Gaussian blur
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                
                # Göz bebeği maskesi
                pupil_mask = np.zeros_like(blurred)
                center = (size * scale // 2, size * scale // 2)
                cv2.circle(pupil_mask, center, pupil_radius * scale, 255, -1)
                
                # Adaptif eşikleme
                binary = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, -2
                )
                
                # Yansımaları tespit et
                reflections = cv2.bitwise_and(binary, binary, mask=pupil_mask)
                
                # Morfolojik işlemler
                kernel = np.ones((2, 2), np.uint8)
                reflections = cv2.morphologyEx(reflections, cv2.MORPH_OPEN, kernel)
                
                # Bağlı bileşenleri bul
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    reflections, connectivity=8
                )
                
                valid_reflections = []
                
                # Her bileşeni kontrol et
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if 2 <= area <= 100:  # Alan kontrolü
                        centroid = centroids[i]
                        dist_to_center = np.sqrt(
                            (centroid[0] - center[0])**2 + (centroid[1] - center[1])**2
                        )
                        
                        if dist_to_center <= pupil_radius * scale * 0.9:  # Merkeze uzaklık kontrolü
                            valid_reflections.append(centroid)
                
                # En fazla 2 yansıma seç
                valid_reflections = sorted(valid_reflections,
                                           key=lambda c: np.sqrt((c[0] - center[0])**2 + (c[1] - center[1])**2)
                                          )[:2]
                
                # Yansımaları maskele
                for centroid in valid_reflections:
                    global_x = x + int(centroid[0] // scale)
                    global_y = y + int(centroid[1] // scale)
                    cv2.circle(mask, (global_x, global_y), 1, 255, -1)
                
                print(f"{eye_type} göz yansıması: {'Bulundu' if valid_reflections else 'Bulunamadı'}")
                
            except Exception as e:
                print(f"Göz işleme hatası ({eye_type}): {str(e)}")
                continue
        
        return mask

def process_all_images(input_dir, output_dir, input_csv_path):
    """
    Belirtilen input_csv dosyasındaki görüntü isimlerini kullanarak,
    her bir görüntü için tespit edilen yansıma maskesini oluşturur.
    Maske dosyaları, 'image_0001_mask.png', 'image_0002_mask.png', ... şeklinde kaydedilir.
    İşlenen görüntülerin dosya isimleri ile maske dosyalarının isimleri output.csv dosyasına kaydedilir.
    """
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # input.csv dosyasını oku
    try:
        df_input = pd.read_csv(input_csv_path)
        # CSV'de görüntü dosyalarının isimlerinin bulunduğu sütun 'file_name' olarak varsayılmıştır.
        files = df_input['file_name'].tolist()
    except Exception as e:
        print(f"Hata: {input_csv_path} dosyası okunamadı: {str(e)}")
        return
    
    total_images = len(files)
    processed_images = 0
    
    # CSV için sonuçları saklamak üzere liste
    results = []
    
    print("Görüntüler işleniyor...")
    
    for idx, filename in enumerate(files, start=1):
        image_path = os.path.join(input_dir, filename)
        
        try:
            # Görüntüyü oku
            image = cv2.imread(image_path)
            if image is None:
                print(f"Hata: {filename} okunamadı!")
                continue
            
            # Göz yansımalarını tespit et
            mask = detect_eye_reflections(image)
            
            # Yeni dosya adını oluştur (örnek: image_0001_mask.png)
            new_filename = f"image_{idx:04d}_mask.png"
            mask_path = os.path.join(output_dir, new_filename)
            
            # Maskeyi kaydet
            cv2.imwrite(mask_path, mask)
            
            processed_images += 1
            print(f"İşlendi ({processed_images}/{total_images}): {filename}")
            
            # CSV için veri ekle: orijinal dosya adı ve oluşturulan maske dosya adı
            results.append([filename, new_filename])
            
        except Exception as e:
            print(f"Hata ({filename}): {str(e)}")
            continue
    
    print("\nİşlem tamamlandı!")
    print(f"Toplam görüntü sayısı: {total_images}")
    print(f"İşlenen görüntü sayısı: {processed_images}")
    print(f"Hatalı görüntü sayısı: {total_images - processed_images}")
    
    # output.csv dosyasını, output klasörünün bir üstünde oluşturuyoruz.
    output_csv = os.path.join(os.path.dirname(output_dir), "output.csv")
    df = pd.DataFrame(results, columns=["file_name", "mask_file_name"])
    df.to_csv(output_csv, index=False)
    print(f"CSV dosyası oluşturuldu: {output_csv}")

def main():
    # Docker ortamında çalışırken varsayılan giriş ve çıkış dizinleri volume olarak tanımlanmıştır.
    default_input_dir = "/input"
    default_output_dir = "/output"
    default_input_csv = os.path.join(default_input_dir, "input.csv")
    
    parser = argparse.ArgumentParser(description="Göz Yansıması Maske Oluşturucu")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=default_input_dir,
        help="Giriş görüntülerinin bulunduğu dizin (varsayılan: %(default)s)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Oluşturulacak maskelerin kaydedileceği dizin (varsayılan: %(default)s)"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=default_input_csv,
        help="İşlenecek görüntülerin isimlerinin bulunduğu CSV dosyası (varsayılan: %(default)s)"
    )
    args = parser.parse_args()
    
    process_all_images(args.input_dir, args.output_dir, args.input_csv)

if __name__ == "__main__":
    main()
