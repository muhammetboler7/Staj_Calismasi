/**
 * mono_tuyz: TUYZ drone görüntülerinde monoküler ORB-SLAM3 çalıştırıp
 * her frame için online kamera pozunu (Twc, TUM formatı) kaydeder.
 *
 * Kullanım: ./mono_tuyz vocab settings.yaml image_list.txt out_traj.txt [viewer:0|1]
 *   image_list.txt satırları: <timestamp> <mutlak_görüntü_yolu>
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <System.h>

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "Usage: ./mono_tuyz vocab settings image_list.txt out_traj.txt [viewer:0|1]" << std::endl;
        return 1;
    }

    std::vector<double> vTimestamps;
    std::vector<std::string> vImagePaths;
    {
        std::ifstream f(argv[3]);
        if (!f)
        {
            std::cerr << "Cannot open image list: " << argv[3] << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(f, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream ss(line);
            double t;
            std::string p;
            ss >> t >> p;
            vTimestamps.push_back(t);
            vImagePaths.push_back(p);
        }
    }
    std::cout << "Loaded " << vImagePaths.size() << " images." << std::endl;

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
            out << vTimestamps[i] << " "
                << t.x() << " " << t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
            nTracked++;
        }
    }
    out.close();

    std::cout << "Tracked frames: " << nTracked << "/" << vImagePaths.size() << std::endl;

    SLAM.Shutdown();
    SLAM.SaveKeyFrameTrajectoryTUM(std::string(argv[4]) + ".keyframes");

    return 0;
}
