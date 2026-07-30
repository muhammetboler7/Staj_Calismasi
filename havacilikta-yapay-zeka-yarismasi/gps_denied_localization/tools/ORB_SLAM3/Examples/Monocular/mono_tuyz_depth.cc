/**
 * mono_tuyz_depth: mono_tuyz'in "medyan sahne derinligi" cikaran varyanti.
 *
 * Deney (Z ekseni): BA reprojeksiyon hatasi optik eksen (derinlik) yonunu
 * zayif kisitladigi icin SLAM'in z POZISYONU drift ediyor (olculdu: online
 * 5.91m vs global-BA 5.96m Z RMSE - BA Z'yi duzeltmiyor). Alternatif sinyal:
 * o karede TAKIP EDILEN harita noktalarinin kamera cercevesindeki MEDYAN
 * derinligi. Nadir bakista bu ~ yerden yukseklik (SLAM biriminde) ve her
 * karede haritadan taze olculur (pozisyon gibi birikerek drift etmez -
 * hipotez bu, bu program o hipotezi olcmek icin veri uretir).
 *
 * Cikti formati mono_tuyz'un TUM formatina IKI ek sutun ekler:
 *   <ts> <tx> <ty> <tz> <qx> <qy> <qz> <qw> <med_depth> <n_tracked>
 *   med_depth: takip edilen noktalarin kamera-z (derinlik) medyani; nokta
 *              yoksa -1 yazilir.
 *   n_tracked: medyana giren nokta sayisi (guven gostergesi).
 *
 * Kullanim: ./mono_tuyz_depth vocab settings.yaml image_list.txt out_traj.txt [viewer:0|1]
 */
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <System.h>
#include <MapPoint.h>

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "Usage: ./mono_tuyz_depth vocab settings image_list.txt out_traj.txt [viewer:0|1]" << std::endl;
        return 1;
    }

    std::vector<double> vTimestamps;
    std::vector<std::string> vImagePaths;
    {
        std::ifstream list(argv[3]);
        if (!list)
        {
            std::cerr << "Cannot open image list: " << argv[3] << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(list, line))
        {
            if (line.empty())
                continue;
            std::istringstream ss(line);
            double ts;
            std::string path;
            ss >> ts >> path;
            vTimestamps.push_back(ts);
            vImagePaths.push_back(path);
        }
    }

    const bool bUseViewer = (argc > 5) ? std::atoi(argv[5]) != 0 : false;
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, bUseViewer);

    std::ofstream out(argv[4]);
    out << std::fixed << std::setprecision(9);

    int nTracked = 0;
    for (size_t i = 0; i < vImagePaths.size(); i++)
    {
        cv::Mat im = cv::imread(vImagePaths[i], cv::IMREAD_UNCHANGED);
        if (im.empty())
        {
            std::cerr << "Failed to load: " << vImagePaths[i] << std::endl;
            continue;
        }

        Sophus::SE3f Tcw = SLAM.TrackMonocular(im, vTimestamps[i]);

        // 2 == Tracking::OK
        if (SLAM.GetTrackingState() == 2)
        {
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f t = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();

            // Takip edilen noktalarin kamera cercevesindeki derinlik medyani.
            // Xc = Rcw * Xw + tcw; derinlik = Xc.z (optik eksen bileseni).
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

            out << vTimestamps[i] << " "
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
                << medDepth << " " << depths.size() << "\n";
            nTracked++;
        }
    }
    out.close();

    std::cout << "Tracked frames: " << nTracked << "/" << vImagePaths.size() << std::endl;

    SLAM.Shutdown();
    SLAM.SaveKeyFrameTrajectoryTUM(std::string(argv[4]) + ".keyframes");

    return 0;
}
