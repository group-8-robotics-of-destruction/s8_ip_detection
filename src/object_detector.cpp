#include <ros/ros.h>
#include <s8_common_node/Node.h>
#include <s8_msgs/DistPose.h>
#include <s8_msgs/isFrontWall.h>
// PCL specific includes
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/surface/mls.h>
//#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
// OTHER
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cstdlib>
#include <vector>
#include <cmath>


// DEFINITIONS
#define HZ                  10
#define BUFFER_SIZE         1

#define NODE_NAME           		"s8_object_detection_node"
#define TOPIC_POINT_CLOUD   		"/camera/depth_registered/points"
#define TOPIC_EXTRACTED_OBJECTS		"/s8/detectedObject"
#define TOPIC_IS_FRONT_WALL         "/s8/isFrontWall"
#define TOPIC_OBJECTS_POS		    "/s8/ip/detection/distPose"
#define CONFIG_DOC                  "/catkin_ws/src/s8_ip_detection/parameters/parameters.json"

// PARAMETERS
#define PARAM_FILTER_X_NAME						"filter_x"
#define PARAM_FILTER_X_DEFAULT					0.1
#define PARAM_FILTER_Z_NAME						"filter_z"
#define PARAM_FILTER_Z_DEFAULT					1.0
#define PARAM_FILTER_Y_NAME						"filter_y"
#define PARAM_FILTER_Y_DEFAULT					0.2
#define PARAM_FLOOR_EXTRACTION_DIST_NAME		"floor_extraction_dist"
#define PARAM_FLOOR_EXTRACTION_DIST_DEFAULT		0.02
#define PARAM_VOXEL_LEAF_SIZE_NAME				"voxel_leaf_size"
#define PARAM_VOXEL_LEAF_SIZE_DEFAULT			0.005
#define PARAM_SEG_DISTANCE_NAME					"seg_distance"
#define PARAM_SEG_DISTANCE_DEFAULT				0.01
#define PARAM_SEG_PERCENTAGE_NAME				"seg_percentage"
#define PARAM_SEG_PERCENTAGE_DEFAULT			0.2
#define PARAM_CAM_ANGLE_NAME				    "cam_angle"
#define PARAM_CAM_ANGLE_DEFAULT			        0.4

typedef pcl::PointXYZRGB PointT;
using namespace std;

class ObjectDetector : public s8::Node
{
    const int hz;

    ros::Subscriber point_cloud_subscriber;
    ros::Publisher point_cloud_publisher;
    ros::Publisher object_distPos_publisher;
    ros::Publisher isFrontWall_publisher;
    ros::Publisher object_cloudcolor_publisher;
    pcl::PointCloud<PointT>::Ptr cloud;

    double filter_x, filter_y, filter_z, max_object_height;
    double floor_extraction_dist;
    double	voxel_leaf_size;
    double 	seg_distance, seg_percentage, normal_distance_weight;
    double cam_angle;

    double H_floor_high;
    double H_floor_low;
    double S_floor_high;
    double S_floor_low;
    double H_wall_high;
    double H_wall_low ;
    double S_wall_high;
    double S_wall_low;

    bool cloudInitialized;

    struct center_of_mass {
        double x_width;
        double y_width;
        double z_width;
        double x_center;
        double y_center;
        double z_center;
        double x_min, x_max;
        double y_min, y_max;
        double z_min, z_max;
    };

public:
    ObjectDetector(int hz) : hz(hz)
    {
        add_params();
        //printParams();
        point_cloud_subscriber   = nh.subscribe(TOPIC_POINT_CLOUD, BUFFER_SIZE, &ObjectDetector::point_cloud_callback, this);
        point_cloud_publisher    = nh.advertise<sensor_msgs::PointCloud2> (TOPIC_EXTRACTED_OBJECTS, BUFFER_SIZE);
        isFrontWall_publisher    = nh.advertise<s8_msgs::isFrontWall> (TOPIC_IS_FRONT_WALL, BUFFER_SIZE);
        object_distPos_publisher = nh.advertise<s8_msgs::DistPose> (TOPIC_OBJECTS_POS, BUFFER_SIZE);
        cloud = pcl::PointCloud<PointT>::Ptr (new pcl::PointCloud<PointT>);
        cloudInitialized = false;
    }

    void updateObject()
    {
        if ( cloudInitialized == false)
            return;
        pcl::PointCloud<PointT>::Ptr cloud_out (new pcl::PointCloud<PointT>);
        passthroughFilterCloud(cloud, 0.3, filter_z, -filter_x, filter_x, -filter_y, 0.4);
        rotatePointCloud(cloud, cloud, -cam_angle);
        *cloud_out = *cloud;
        voxelGridCloud(cloud);
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        //extractFloor(cloud, coefficients);
        //if (coefficients->values.size() < 2)
        //	return;
        //double theta = getAngle(coefficients);
        //cout << "theta: " << theta << endl;
        cloud = removeWallsCloud(cloud);
        //passthroughFilterCloud(cloud, 0.3, filter_z, -filter_x, filter_x, max_object_height, 0.40);
        if (cloud->points.size() != 0)
            clusterCloud(cloud, cloud_out);

        //Eigen::Vector3f mass_center;
        //getMoments(cloud, &mass_center);
        //cout << "x: " << mass_center(0) << " y: " << mass_center(1) << " z: " << mass_center(2) << endl;

        //cloudPublish(cloud);
    }

private:
    // Removes outliers using a StatisticalOutlierRemoval filter
    void statisticalOutlierRemovalCloud(pcl::PointCloud<PointT>::Ptr cloud_stat)
    {
        pcl::StatisticalOutlierRemoval<PointT> sta;
        sta.setInputCloud (cloud_stat);
        sta.setMeanK (20);
        sta.setStddevMulThresh (1.0);
        sta.filter (*cloud_stat);
    }

    // Calculates the angle between two 3D planes.
    double getAngle (pcl::ModelCoefficients::Ptr coefficients)
    {

        double q0 = coefficients->values[0];
        double q1 = coefficients->values[1];
        double q2 = coefficients->values[2];

        float dotproduct 	= q2;
        float len1 			= 1;
        float len2 			= q0*q0+q1*q1+q2*q2;
        float angleRad 		= acos(dotproduct/(sqrt(len1)*sqrt(len2)));
        return M_PI/2-angleRad;
    }

    bool getAngleToWall(pcl::ModelCoefficients::Ptr coefficients)
    {

        double qx = coefficients->values[0];
        double qy = coefficients->values[1];
        double qz = coefficients->values[2];

        double wx = 1;
        double wy = 1;
        double wz = 0;

        double length = std::abs(qx*wx + qy*wy);
        double q_euclidean = std::sqrt(qx*qx + qy*qy + qz*qz);
        double w_euclidean = std::sqrt(2); 
        double angle = std::acos(length / (q_euclidean * w_euclidean));
        //ROS_INFO("angle %lf", angle);


        return (std::abs(angle)-1.56 <= 0.2 && std::abs(angle)-1.56 >= -0.2);
    }

    // void getMoments(pcl::PointCloud<PointT>::Ptr inputCloud, Eigen::Vector3f *mass_center, PointT *min_point, PointT *max_point)
    // {
    //     pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
    //     feature_extractor.setInputCloud (inputCloud);
    //     feature_extractor.compute ();
    //     feature_extractor.getMassCenter (*mass_center);
    //     feature_extractor.getAABB (*min_point, *max_point);
    // }

    void rotatePointCloud(pcl::PointCloud<PointT>::Ptr inputCloud, pcl::PointCloud<PointT>::Ptr outputCloud, double theta)
    {
        // Create rotation matrix.
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitX()));

        // Transform the cloud and return it
        pcl::transformPointCloud (*inputCloud, *outputCloud, transform);;
    }

    // Down samples the point cloud using VoxelGrid filter
    // to make computations easier.
    void voxelGridCloud(pcl::PointCloud<PointT>::Ptr cloud_stat)
    {
        pcl::VoxelGrid<PointT> sor;
        sor.setInputCloud (cloud_stat);
        sor.setLeafSize ((float)voxel_leaf_size, (float)voxel_leaf_size, (float)voxel_leaf_size);
        sor.filter (*cloud_stat);
    }

    // Build a passthrough filter to reduce field of view.
    void passthroughFilterCloud(pcl::PointCloud<PointT>::Ptr cloud_filter, double zmin, double zmax, double xmin, double xmax, double ymin, double ymax)
    {
        pcl::PassThrough<PointT> pass;

        pass.setInputCloud (cloud_filter);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (zmin, zmax);
        pass.filter (*cloud_filter);

        pass.setInputCloud (cloud_filter);
        pass.setFilterFieldName ("x");
        pass.setFilterLimits (xmin, xmax);
        pass.filter (*cloud_filter);

        pass.setInputCloud (cloud_filter);
        pass.setFilterFieldName ("y");
        pass.setFilterLimits (ymin, ymax);
        pass.filter (*cloud_filter);
    }

    void extractFloor(pcl::PointCloud<PointT>::Ptr cloud_floor, pcl::ModelCoefficients::Ptr coeff)
    {
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

        // Create the segmentation object that segments the biggest
        // plane with in 45 degree deviation from the z axis
        // and gets the planes coefficients.
        pcl::SACSegmentation<PointT> seg;
        seg.setModelType (pcl::SACMODEL_PARALLEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setMaxIterations(100);
        seg.setDistanceThreshold (floor_extraction_dist);
        seg.setAxis(Eigen::Vector3f(1.0,0.0,1.0));
        seg.setEpsAngle (M_PI/4);
        seg.setInputCloud (cloud_floor);
        seg.segment (*inliers, *coeff);

        // create the extraction objects that removes the floor
        // from the cloud based on the segmentation.
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (cloud_floor);
        extract.setIndices (inliers);
        extract.setNegative (true);
        extract.filter (*cloud_floor);
    }

    pcl::PointCloud<PointT>::Ptr removeWallsCloud(pcl::PointCloud<PointT>::Ptr cloud_seg)
    {
        pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT>);
        pcl::ModelCoefficients::Ptr coeff (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
        pcl::NormalEstimation<PointT, pcl::Normal> ne;

        pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
        pcl::ExtractIndices<PointT> extract;

        // Estimate point normals
        ne.setSearchMethod (tree);
        ne.setKSearch (50);

        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (seg_distance);
        seg.setNormalDistanceWeight (normal_distance_weight);
        seg.setMaxIterations (1000);

        int i = 0, nr_points = (int) cloud_seg->points.size ();
        // While 20% of the original cloud is still there
        while (cloud_seg->points.size () > seg_percentage * nr_points && i < 10 && cloud_seg->points.size() > 0)
        {
            ROS_INFO("IN THE WHILE");
            //seg.setInputCloud (cloud);
            ne.setInputCloud (cloud_seg);
            ne.compute (*cloud_normals);
            //seg.setInputCloud (cloud);
            seg.setInputCloud (cloud_seg);
            seg.setInputNormals (cloud_normals);
            seg.segment (*inliers, *coeff);
            ROS_INFO("GOT INLIERS AND COEFFICIENTS");
            if (inliers->indices.size () == 0)
            {
                break;
            }
            if(inliers->indices.size() < nr_points/20 || inliers->indices.size() < 10){
                i++;
                continue;
            }
            // Extract the planar inliers from the input cloud
            ROS_INFO("TRYING TO EXTRACT");
            extract.setInputCloud (cloud_seg);
            extract.setIndices (inliers);
            extract.setNegative (true);
            extract.filter (*cloud_plane);
            ROS_INFO("FILTERED THE CLOUD");
            cloud_seg.swap (cloud_plane);
            i++;
        }
        return cloud_seg;
    }

    void clusterCloud(pcl::PointCloud<PointT>::Ptr cloud_input, pcl::PointCloud<PointT>::Ptr cloud_out)
    {

        bool isWall = false;
        double wallDistance = 0.0;
        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
        tree->setInputCloud (cloud_input);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance (0.02); // 2cm
        ec.setMinClusterSize (50);
        ec.setMaxClusterSize (25000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (cloud_input);
        ec.extract (cluster_indices);

        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
        {
            pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
            for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
            {
                cloud_cluster->points.push_back (cloud_input->points[*pit]); //*
            }

            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            cloud_cluster->header.frame_id = "camera_rgb_frame";

            statisticalOutlierRemovalCloud(cloud_cluster);
            center_of_mass massCenter;
            getCloudSize(cloud_cluster, &massCenter);
            //ROS_INFO("X Width: %lf, Z Width: %lf, Z Width: %lf, Center of Mass: %lf", massCenter.x_width, massCenter.y_width, massCenter.z_width, massCenter.z_center);
            if(massCenter.x_width > 0.01 && massCenter.x_width < 0.10 && massCenter.y_width > 0.01 && massCenter.y_width < 0.10 && massCenter.z_width < 0.10)// && massCenter.y_center > 0.15 )
            {
                float H = 0.0, S = 0.0, V = 0.0;
                getColors(cloud_cluster, &H, &S, &V);
                ROS_INFO("H: %lf, S: %lf, V: %lf", H, S, V);
                if (H < 0.18 && H > 0.08 && S < 0.45)
                {
                    ROS_INFO("Probably a wall");
                }
                else
                {
                    //passthroughFilterCloud(cloud_out, massCenter.z_min, massCenter.z_max, massCenter.x_min, massCenter.x_max, massCenter.y_min, massCenter.y_max);
                    cloudPublish(cloud_cluster);
                    distPosePublish(massCenter.x_center, massCenter.z_center);
                    break;
                }
			}
            else if(massCenter.y_width > 0.10 && massCenter.z_width < 0.04 && massCenter.x_center > -0.11 && massCenter.x_center < 0.13 && massCenter.z_center < 0.40){
                ROS_INFO("PROBABLY A WALL IN FRONT, x size %lf", massCenter.x_width);
                isWall = true;
                wallDistance = massCenter.z_center;
            }
            else{
                ROS_INFO("SIZE INCORRECT");
            }
            j++;
            wallPublish(isWall, wallDistance);
        }

    }

    void getCloudSize(pcl::PointCloud<PointT>::Ptr cloud_moment, center_of_mass *center)
    {
        int cloud_size = cloud_moment->points.size();

        double x_min = cloud_moment->points[0].x;
        double x_max = cloud_moment->points[0].x;
        double y_min = cloud_moment->points[0].y;
        double y_max = cloud_moment->points[0].y;
        double z_min = cloud_moment->points[0].z;
        double z_max = cloud_moment->points[0].z;
        double x_avg = 0, y_avg = 0, z_avg = 0;

        // Loop through all points to find size
        for(int iter = 1; iter != cloud_moment->points.size(); ++iter)
        {
            double x_tmp = cloud_moment->points[iter].x;
            double y_tmp = cloud_moment->points[iter].y;
            double z_tmp = cloud_moment->points[iter].z;
            if (x_tmp > x_max)
                x_max = x_tmp;
            if (x_tmp < x_min)
                x_min = x_tmp;
            if (y_tmp > y_max)
                y_max = y_tmp;
            if (y_tmp < y_min)
                y_min = y_tmp;
            if (z_tmp > z_max)
                z_max = z_tmp;
            if (z_tmp < z_min)
                z_min = z_tmp;
            x_avg += x_tmp;
            y_avg += y_tmp;
            z_avg += z_tmp;
        }
        (*center).x_center = x_avg / cloud_size;
        (*center).y_center = y_avg / cloud_size;
        (*center).z_center = z_avg / cloud_size;
        (*center).x_width = std::abs(x_max - x_min);
        (*center).y_width = std::abs(y_max - y_min);
        (*center).z_width = std::abs(z_max - z_min);
        (*center).x_min = x_min;
        (*center).x_max = x_max;
        (*center).y_min = y_min;
        (*center).y_max = y_max;
        (*center).z_min = z_min;
        (*center).z_max = z_max;
    }

    void getColors(pcl::PointCloud<PointT>::Ptr cloud_color, float *H, float *S, float *V)
    {
        double R_avg = 0;
        double G_avg = 0;
        double B_avg = 0;
        double R_acc = 0;
        double G_acc = 0;
        double B_acc = 0;
        float H_acc = 0;
        float S_acc = 0;
        float V_acc = 0;
        for(int iter = 0; iter != cloud_color->points.size(); ++iter)
        {
            //Whatever you want to do with the points
            uint8_t R = cloud_color->points[iter].r;
            uint8_t G = cloud_color->points[iter].g;
            uint8_t B = cloud_color->points[iter].b;
            //ROS_INFO("R:, %d G: %d B: %d iter, %d", R, G, B, iter);
            R_acc += R;
            G_acc += G;
            B_acc += B;
        }
        //ROS_INFO("R_low: %d R_high: %d G_low: %d G_high: %d B_low: %d B_high: %d", R_low, R_high, G_low, R_high, B_low, B_high);
        int size = cloud_color->points.size();
        R_avg = R_acc / size;
        G_avg = G_acc / size;
        B_avg = B_acc / size;
        RGB2HSV((float)R_avg, (float)G_avg, (float)B_avg, *H, *S, *V);
        ROS_INFO("H: %f, S: %f, V: %f", *H, *S, *V);
    }

    static void RGB2HSV(float r, float g, float b, float &h, float &s, float &v)
    {
        float K = 0.f;

        if (g < b)
        {
            std::swap(g, b);
            K = -1.f;
        }

        if (r < g)
        {
            std::swap(r, g);
            K = -2.f / 6.f - K;
        }

        float chroma = r - std::min(g, b);
        h = fabs(K + (g - b) / (6.f * chroma + 1e-20f));
        s = chroma / (r + 1e-20f);
        v = r/255;
    }


    void point_cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        pcl::fromROSMsg (*cloud_msg, *cloud);
        cloudInitialized = true;
    }

    void cloudPublish(pcl::PointCloud<PointT>::Ptr cloud_pub)
    {
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*cloud_pub, output);
        point_cloud_publisher.publish (output);
    }

    void distPosePublish(double x, double z)
    {
        s8_msgs::DistPose distPose;
        double theta = std::atan(x/z);
        double dist  = std::sqrt(x*x+z*z);

        distPose.pose = (float)theta;
        distPose.dist = (float)dist;
        object_distPos_publisher.publish(distPose);
    }

    void wallPublish(bool isWall, double distance){
        s8_msgs::isFrontWall frontWall;
        frontWall.isFrontWall = isWall;
        frontWall.distToFrontWall = distance;

        isFrontWall_publisher.publish(frontWall);
    }

    void add_params()
    {
        std::string home(::getenv("HOME"));
        ROS_INFO("home: %s", CONFIG_DOC);
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(home + CONFIG_DOC, pt);
        // CIRCLE
        filter_x = pt.get<double>("filter_x");
        filter_y = pt.get<double>("filter_y");
        filter_z = pt.get<double>("filter_z");
        max_object_height = pt.get<double>("max_object_height");
        floor_extraction_dist = pt.get<double>("floor_extraction_dist");
        voxel_leaf_size = pt.get<double>("voxel_leaf_size");
        seg_distance = pt.get<double>("seg_distance");
        seg_percentage = pt.get<double>("seg_percentage");
        normal_distance_weight = pt.get<double>("normal_distance_weight");

        H_floor_high = pt.get<double>("H_floor_high");
        H_floor_low = pt.get<double>("H_floor_low");
        S_floor_high = pt.get<double>("S_floor_high");
        S_floor_low = pt.get<double>("S_floor_low");
        H_wall_high = pt.get<double>("H_wall_high");
        H_wall_low  = pt.get<double>("H_wall_low");
        S_wall_high = pt.get<double>("S_wall_high");
        S_wall_low = pt.get<double>("S_wall_low");

        cam_angle = pt.get<double>("cam_angle");
    }
};

int main(int argc, char **argv) {

    ros::init(argc, argv, NODE_NAME);

    ObjectDetector detector(HZ);
    ros::Rate loop_rate(HZ);

    while(ros::ok()) {
        ros::spinOnce();
        detector.updateObject();
        loop_rate.sleep();
    }

    return 0;
}
