/**
 * mono_tuyz_stream: mono_tuyz'in canli/streaming varyanti.
 *
 * mono_tuyz sabit bir image_list.txt'yi baştan sona işleyip sonda tek
 * dosyaya yazan bir batch programdır (bkz. mono_tuyz.cc). Canlı yarışma
 * sunucusundan frame'ler HTTP ile teker teker geldiği için ORB-SLAM3
 * sürecinin (ve haritasının) oturum boyunca AYAKTA kalması, her yeni
 * frame geldiğinde tek bir TrackMonocular() çağrısıyla beslenmesi ve
 * pozun HEMEN geri okunabilmesi gerekir. Bu program bunu stdin (komut)
 * + ayrı bir pipe fd (protokol çıktısı) üzerinden satır-satır bir
 * protokolle yapar; Python tarafı (bkz.
 * TAKIM_BAGLANTI_ARAYUZU/src/orbslam_stream.py) bu süreci başlatıp her
 * frame için stdin'e bir satır yazar, protokol pipe'ından bir satır okur.
 *
 * ÖNEMLİ: ORB-SLAM3'ün kendi iç kodu (Tracking/LocalMapping/System vs.)
 * durum mesajlarını doğrudan std::cout'a yazıyor (ör. "New Map created
 * with N points"), rastgele anlarda, bizim protokol satırlarımızla aynı
 * kanala karışabiliyor. Bu yüzden protokol çıktısı stdout'ta DEĞİL, ayrı
 * bir dosya tanımlayıcısında (--proto-fd ile verilir) yazılır; stdout
 * ORB-SLAM3'ün kendi gürültülü loglarına bırakılır.
 *
 * Kullanım: ./mono_tuyz_stream vocab settings.yaml --proto-fd=<N>
 *   <N>: ebeveyn sürecin bu çocuğa miras bıraktığı, yazma ucu açık bir
 *   pipe'ın dosya tanımlayıcı numarası (bkz. orbslam_stream.py).
 *
 * Protokol:
 *   stdin      : "<frame_id> <mutlak_goruntu_yolu>"    veya   "QUIT"
 *   proto-fd   : "<frame_id> OK <tx> <ty> <tz> <qx> <qy> <qz> <qw> <med_depth> <n_tracked>\n"
 *                "<frame_id> FAIL\n"                    (tracking kaybedildi)
 *                "<frame_id> ERR <mesaj>\n"              (goruntu okunamadi vb.)
 *                Program hazır olduğunda ilk satır olarak "READY\n" yazılır.
 *
 *   med_depth: o karede takip edilen harita noktalarinin kamera cercevesindeki
 *   derinlik (optik eksen bileseni) MEDYANI; nokta yoksa -1. n_tracked: medyana
 *   giren nokta sayisi. Gorev 2 Z ekseni icin eklendi: SLAM'in z POZISYONU
 *   birikimli drift tasir ve BA bunu duzeltemez (olculdu: online 5.91m vs
 *   global-BA 5.96m Z RMSE); medyan derinlik ise her karede haritadan taze
 *   olculur ve Z icin daha iyi sinyaldir (bkz. TAKIM_BAGLANTI_ARAYUZU/
 *   z_ekseni_deneyleri_aciklama.md ve online_alignment.py "depth_blend").
 * stdin EOF (sunucu tarafi pipe'i kapatirsa) de QUIT ile aynı şekilde çıkışa
 * neden olur.
 */
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <System.h>
#include <MapPoint.h>

static int g_protoFd = -1;

static void proto_write(const std::string &s)
{
    std::string line = s + "\n";
    ssize_t off = 0;
    while (off < static_cast<ssize_t>(line.size()))
    {
        ssize_t n = write(g_protoFd, line.data() + off, line.size() - off);
        if (n <= 0)
            break;
        off += n;
    }
}

int main(int argc, char **argv)
{
    if (argc < 4 || strncmp(argv[3], "--proto-fd=", 11) != 0)
    {
        std::cerr << "Usage: ./mono_tuyz_stream vocab settings.yaml --proto-fd=<N>" << std::endl;
        return 1;
    }
    g_protoFd = std::atoi(argv[3] + 11);

    // Viewer'siz: canli sunucu kosumunda ekransiz (headless) calisir.
    // NOT: System() constructor'i ve TrackMonocular() std::cout'a durum
    // mesajlari basar; bunlari kasitli olarak yoksuyoruz (bkz. yukaridaki
    // aciklama) - protokol cikisi g_protoFd uzerinden gider.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, false);

    proto_write("READY");

    std::ostringstream fmt;
    fmt << std::fixed << std::setprecision(9);

    std::string line;
    long frameCounter = 0;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
            continue;
        if (line == "QUIT")
            break;

        std::istringstream ss(line);
        std::string frameId, imagePath;
        ss >> frameId >> imagePath;
        if (frameId.empty() || imagePath.empty())
            continue;

        cv::Mat im = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (im.empty())
        {
            proto_write(frameId + " ERR cannot_read_image");
            continue;
        }

        // Gercek zaman damgasi onemli degil (IMU kullanilmiyor); tekdüze
        // artan bir sayac SLAM icin yeterli ve sirali garanti eder.
        double timestamp = static_cast<double>(frameCounter++) / 30.0;

        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, timestamp);

        // 2 == Tracking::OK
        if (SLAM.GetTrackingState() == 2)
        {
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f t = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();

            // Takip edilen noktalarin kamera-cercevesi derinlik medyani
            // (Xc = Tcw * Xw; derinlik = Xc.z). Medyan: tek tuk hatali
            // nokta sonucu bozmasin diye.
            std::vector<float> depths;
            for (ORB_SLAM3::MapPoint *pMP : SLAM.GetTrackedMapPoints())
            {
                if (!pMP || pMP->isBad())
                    continue;
                Eigen::Vector3f Xc = Tcw * pMP->GetWorldPos();
                if (Xc.z() > 0)
                    depths.push_back(Xc.z());
            }
            float medDepth = -1.0f;
            if (!depths.empty())
            {
                size_t mid = depths.size() / 2;
                std::nth_element(depths.begin(), depths.begin() + mid, depths.end());
                medDepth = depths[mid];
            }

            fmt.str("");
            fmt << frameId << " OK "
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                << medDepth << " " << depths.size();
            proto_write(fmt.str());
        }
        else
        {
            proto_write(frameId + " FAIL");
        }
    }

    SLAM.Shutdown();
    return 0;
}
