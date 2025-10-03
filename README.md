# ** İnsan göz bebeğine yansıyan ışık analiz/tespit edilmesi**

Proje, **Python** ve **OpenCV** kullanarak İnsan göz bebeğine yansıyan ışık tespit eden eden bir sistemdir.

---

## **📌 1. Gereksinimler ve Kurulum**

Proje, aşağıdaki Python bağımlılıklarına ihtiyaç duymaktadır:

| 🔧 Bileşen           | 🏷️ Sürüm |
| -------------------- | -------- |
| **CUDA Sürümü**      | 12.0     |
| **Python Sürümü**    | 3.8+     |
| **MediaPipe Sürümü** | 0.10.0   |
| **Pillow Sürümü**    | 10.0.0   |
| **Pandas Sürümü**    | 1.3.0    |
| **OpenCV Sürümü**    | 4.5.0    |

## **🐳 2. Docker Kullanımı**

Proje, Docker konteyneri içerisinde çalıştırılmak üzere yapılandırılmıştır.
Bu, test ortamında standartlaştırılmış bir çalışma ortamı sağlar.

### **📌 2.1. Docker İmajı Oluşturma**

Docker imajını oluşturmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
sudo docker build -f docker/Dockerfile -t deepfake-photo-analysis .
```

komut, mevcut Dockerfile kullanarak gerekli bağımlılıkları içeren bir Docker imajı oluşturacaktır.

### **📌 2.2. Docker Konteynerini Çalıştırma**

Oluşturulan imajı bir Docker konteyneri içinde çalıştırmak için:

```bash
sudo docker run --gpus all \
    -v /path/to/input/:/input \ # input olarak verilecek dosyanın yolu (Not: input olarak verilecek CSV dosyası ve görüntüler ile aynı dizinde bulunmalıdır.)
    -v /path/to/output:/output \ # output.csv dosyasının kaydedileceği local (HOST) dizin
    deepfake-photo-analysis
```

Burada:

- `/path/to/input/`: işlenecek görüntüler ve input.csv dosyasının bulunduğu klasör
- `/path/to/output`: oluşturulacak çıktı dosyasının kaydedileceği klasör olmalıdır.

**Input.csv dosyası ve görüntüler aynı input klasöründe bulunmalıdır!**

## **⚠️3. Docker Kullanımında Dikkat Edilmesi Gerekenler**

- Çıktı klasörü Docker içinde oluşturulacak ve **bağımlı volume olarak tanımlanmalıdır**.
- Çıktı dizinine Docker dışından erişim için izinleri düzenleyin:

```bash
sudo chown -R $USER:$USER /path/to/output/
```

## **📂 4. Klasör Yapısı**

```bash
│── 📂 proje_dizini
│   │── 📂 src
│   │   ├── main.py
│   │── 📂 docker
│   │   ├── Dockerfile
│   │── 📂 dataset
│   │   ├── dataset.csv
│   │   ├── image_0001.png
│   │   ├── image_0002.png
│   │   ├── ...
│   │── README.md
```
