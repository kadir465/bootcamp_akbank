
###Bootcamp Akbank
#Akbank Derin Öğrenme Bootcamp: Meme Kanseri Histopatolojik Görüntü Sınıflandırması
#Proje Özeti
Bu proje, Akbank Derin Öğrenme Bootcamp kapsamında geliştirilmiş bir meme kanseri histopatolojik görüntü sınıflandırma sistemidir. BreaKHis veri seti üzerinde Convolutional Neural Network (CNN) mimarisi kullanılarak benign (iyi huylu) ve malignant (kötü huylu) dokuların otomatik sınıflandırılması amaçlanmıştır. Proje, derin öğrenme modellerinin tıbbi görüntü analizindeki uygulanabilirliğini ve explainable AI tekniklerinin klinik karar destek sistemlerindeki önemini vurgulamaktadır.

#Proje Amacı
Meme kanseri, dünyada kadınlarda en sık görülen kanser türü olup erken teşhis hayat kurtarıcı öneme sahiptir. Geleneksel histopatolojik incelemeler uzman bağımlılığı, subjektif değerlendirmeler ve zaman alıcı süreçler içermektedir. Bu projenin temel amacı, patologların iş yükünü hafifletecek, standartize edilmiş ve yüksek doğruluklu bir otomatik tanı destek sistemi geliştirmektir.

#Veri Seti
Çalışmada BreaKHis (Breast Cancer Histopathological Database) veri seti kullanılmıştır. Veri seti 82 hastadan alınan 7,909 histopatolojik görüntü içermektedir. Görüntüler 40X, 100X, 200X ve 400X büyütme oranlarında olup 700x460 piksel çözünürlüğe sahiptir. Veri seti 2,480 benign ve 5,429 malignant görüntüden oluşmaktadır. Benign kategoride adenosis, fibroadenoma, phyllodes tümör ve tübüler adenoma; malignant kategoride duktal karsinom, lobuler karsinom, müsinöz karsinom ve papiller karsinom alt tipleri bulunmaktadır.

#Yöntem
Teknolojik Altyapı
Projede Python programlama dili ve TensorFlow derin öğrenme kütüphanesi kullanılmıştır. Veri işleme için Pandas ve NumPy, görselleştirme için Matplotlib ve Seaborn kütüphanelerinden yararlanılmıştır. Hiperparametre optimizasyonu için Keras Tuner kullanılmıştır.

#Model Mimarisi
Sequential CNN mimarisi kullanılarak iki konvolüsyon katmanı, max pooling katmanları, batch normalizasyon ve dropout katmanlarından oluşan bir model geliştirilmiştir. Modelin çıkış katmanında sigmoid aktivasyon fonksiyonu ile binary sınıflandırma yapılmıştır.

#Veri Ön İşleme
Görüntüler 64x64 piksel boyutuna yeniden ölçeklendirilmiş ve piksel değerleri 0-1 aralığına normalize edilmiştir. Veri artırma teknikleri kullanılarak modelin generalize yeteneği geliştirilmiştir. Rotation, shift, zoom, flip ve brightness adjustment gibi dönüşümler uygulanmıştır.

#Hiperparametre Optimizasyonu
Random Search yöntemi ile hiperparametre optimizasyonu gerçekleştirilmiştir. Katman sayısı, filtre sayısı, öğrenme oranı, optimizer tipi ve dropout oranı gibi parametreler optimize edilmiştir.

#Sonuçlar
Model validation veri seti üzerinde %85.3 doğruluk, %87.1 kesinlik ve %83.5 duyarlılık değerlerine ulaşmıştır. Grad-CAM (Gradient-weighted Class Activation Mapping) tekniği ile modelin karar mekanizması görselleştirilmiş ve modelin malignant sınıflandırmasında tümör hücre kümelerine odaklandığı gözlemlenmiştir. Örnek bir test görüntüsü üzerinde %92 güven oranıyla malignant tahmini yapılmıştır.

#Katkılar ve Yenilikler
Bu çalışma, derin öğrenme tabanlı sistemlerin tıbbi görüntü analizindeki potansiyelini göstermektedir. Grad-CAM görselleştirmesi ile model kararlarının interpretable hale getirilmesi, klinik uygulamalarda güven artırıcı bir faktör olarak öne çıkmaktadır. Proje, sınırlı veri ile yüksek performanslı model geliştirme stratejileri ve explainable AI'nın tıbbi tanı sistemlerindeki uygulanabilirliği açısından önemli katkılar sunmaktadır.

#Kullanım
Proje dosyaları Jupyter Notebook formatında sunulmuştur. Temel bağımlılıkların yüklenmesinin ardından notebook'lar çalıştırılabilir. Model eğitimi, hiperparametre optimizasyonu ve Grad-CAM görselleştirmesi için ayrı bölümler bulunmaktadır. Veri setine Kaggle üzerinden erişim sağlanabilmektedir.


#Bağlantılar
Veri Seti Adresi: https://www.kaggle.com/datasets/ambarish/breakhis

Kaggle Adresi: https://www.kaggle.com/code/yucelay/akbank-bootcamp1
