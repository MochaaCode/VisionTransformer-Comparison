# Vision Transformer Comparison: ViT vs Swin Transformer

## 📌 Overview

Kode ini membandingkan performa Vision Transformer (ViT) dan Swin Transformer pada dataset CIFAR-10 menggunakan Google Colab dengan GPU.

## ⚙️ Cara Menjalankan di Google Colab

### Langkah 1: Siapkan Google Drive

1. Buat folder di Google Drive dengan nama: `VisionTransformer-Comparison`
2. Di dalam folder tersebut, buat dua subfolder:
   - `results/` (untuk menyimpan hasil)
   - `models/` (untuk menyimpan model weights)

### Langkah 2: Buat Notebook Baru di Colab

1. Buka [Google Colab](https://colab.research.google.com/)
2. Buat notebook baru
3. Ubah runtime ke GPU: `Runtime > Change runtime type > Hardware accelerator > GPU`

### Langkah 3: Copy-Paste Kode

1. Copy seluruh isi file `main.py` di atas
2. Paste ke cell pertama di Colab notebook
3. Jalankan cell tersebut (tekan `Shift + Enter`)

### Langkah 4: Ikuti Petunjuk

- Saat pertama kali dijalankan, akan muncul link untuk menghubungkan Google Drive
- Klik link tersebut, salin kode verifikasi, dan paste di Colab
- Tunggu hingga proses training dan evaluasi selesai (sekitar 30-45 menit)

## 📦 Hasil yang Dihasilkan

Setelah selesai, file-file berikut akan tersimpan di Google Drive:

- `results/training_curves_vit.png` dan `training_curves_swin.png`
- `results/confusion_matrix_vit.png` dan `confusion_matrix_swin.png`
- `results/metrics_comparison.csv`
- `results/parameters_comparison.csv`
- `models/vit_best.pth` dan `models/swin_best.pth`

## 💻 Requirements

- Python 3.8+
- Google Colab dengan GPU Tesla T4/V100
- Akses internet untuk download pre-trained models

## 📋 Catatan Penting

1. **Jangan hapus cell mount Google Drive** - file hasil akan disimpan di sana
2. **Pastikan runtime menggunakan GPU** - klik `Runtime > Change runtime type`
3. **Jika error saat download dataset**, jalankan ulang cell tersebut
4. **Untuk reproduksi**, cukup jalankan ulang seluruh notebook dari awal

## 📞 Bantuan

Jika mengalami masalah saat menjalankan kode, hubungi:

- Nama: [NAMA LU]
- NIM: [NIM LU]
- Email: [EMAIL LU]
