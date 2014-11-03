#include <ros/ros.h>
#include <s8_common_node/Node.h>
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
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
// OTHER
#include <vector>


// DEFINITIONS
#define HZ                  10
#define BUFFER_SIZE         1

#define NODE_NAME           		"s8_object_detection_node"
#define TOPIC_POINT_CLOUD   		"/camera/depth_registered/points"
#define TOPIC_EXTRACTED_OBJECTS		"/s8/detectedObject"

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
	pcl::PointCloud<PointT>::Ptr cloud;

	double filter_x, filter_y, filter_z;
	double floor_extraction_dist;
	double	voxel_leaf_size;
	double 	seg_distance, seg_percentage;
	double cam_angle;

	bool cloudInitialized;

public:
	ObjectDetector(int hz) : hz(hz)
	{
		add_params();
		//printParams();
		point_cloud_subscriber = nh.subscribe(TOPIC_POINT_CLOUD, BUFFER_SIZE, &ObjectDetector::point_cloud_callback, this);
		point_cloud_publisher  = nh.advertise<sensor_msgs::PointCloud2> (TOPIC_EXTRACTED_OBJECTS, BUFFER_SIZE);

		cloud = pcl::PointCloud<PointT>::Ptr (new pcl::PointCloud<PointT>);
		cloudInitialized = false;
	}

	void updateObject()
	{	
		if ( cloudInitialized == false)
			return;
		passthroughFilterCloud(cloud, 0.3, filter_z, -filter_x, filter_x, -filter_y, 0.4);
		voxelGridCloud(cloud);
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		//extractFloor(cloud, coefficients);
		//if (coefficients->values.size() < 2)
		//	return;
		//double theta = getAngle(coefficients);
		//cout << "theta: " << theta << endl;
		cloud = removeWallsCloud(cloud);
		rotatePointCloud(cloud, cloud, -cam_angle);
		passthroughFilterCloud(cloud, 0.3, filter_z, -filter_x, filter_x, 0.15, 0.40);
		clusterCloud(cloud);

		//Eigen::Vector3f mass_center;
		//getMoments(cloud, &mass_center);
		//cout << "x: " << mass_center(0) << " y: " << mass_center(1) << " z: " << mass_center(2) << endl;

		//cloudPublish(cloud);
	}

private:
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

	void getMoments(pcl::PointCloud<PointT>::Ptr inputCloud, Eigen::Vector3f *mass_center, PointT *min_point, PointT *max_point)
	{
		pcl::MomentOfInertiaEstimation <PointT> feature_extractor;
		feature_extractor.setInputCloud (inputCloud);
		feature_extractor.compute ();
		feature_extractor.getMassCenter (*mass_center);
		feature_extractor.getAABB (*min_point, *max_point);
	}

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

		pcl::SACSegmentation<PointT> seg;
		pcl::ExtractIndices<PointT> extract;

		seg.setOptimizeCoefficients (true);
		seg.setModelType (pcl::SACMODEL_PLANE);
		seg.setMethodType (pcl::SAC_RANSAC);
		seg.setDistanceThreshold (seg_distance);

		int i = 0, nr_points = (int) cloud_seg->points.size ();
		// While 20% of the original cloud is still there
		while (cloud_seg->points.size () > seg_percentage * nr_points && i < 10)
		{
			//seg.setInputCloud (cloud);
			seg.setInputCloud (cloud_seg);
			seg.segment (*inliers, *coeff); 
			if (inliers->indices.size () == 0)
			{
				break;
			}
			if(inliers->indices.size() < nr_points/20){
				i++;

				continue;
			}			
			// Extract the planar inliers from the input cloud
			extract.setInputCloud (cloud_seg);
			extract.setIndices (inliers);
			extract.setNegative (true);
			extract.filter (*cloud_plane);
			cloud_seg.swap (cloud_plane);
			i++;
		}
		return cloud_seg;
    }

    void clusterCloud(pcl::PointCloud<PointT>::Ptr cloud_input)
    {
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

			PointT min_point_AABB;
			PointT max_point_AABB;
			Eigen::Vector3f mass_center;
			getMoments(cloud_cluster, &mass_center, &min_point_AABB, &max_point_AABB);
			double xDiff = max_point_AABB.x - min_point_AABB.x;
			double yDiff = max_point_AABB.y - min_point_AABB.y;
			double zDiff = max_point_AABB.z - min_point_AABB.z;
			// Force the cluster to be roughly the shape of the object.
			if (xDiff > 0.02 && xDiff < 0.06 && yDiff > 0.02 && yDiff < 0.06 && zDiff < 0.04)
			{
				cout << "x: " << mass_center(0) << " y: " << mass_center(1) << " z: " << mass_center(2) << endl;
				cloudPublish(cloud_cluster);
			}
			j++;
			cout <<"j: " << j << endl;
		}

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

	void add_params() 
    {
    	// Passthrough filter parameters.
        add_param(PARAM_FILTER_X_NAME, filter_x, PARAM_FILTER_X_DEFAULT);
        add_param(PARAM_FILTER_Y_NAME, filter_y, PARAM_FILTER_Y_DEFAULT);
        add_param(PARAM_FILTER_Z_NAME, filter_z, PARAM_FILTER_Z_DEFAULT);
        // Voxel filtering parameters
        add_param(PARAM_VOXEL_LEAF_SIZE_NAME, voxel_leaf_size, PARAM_VOXEL_LEAF_SIZE_DEFAULT);
    	// Floor extraction.
        add_param(PARAM_FLOOR_EXTRACTION_DIST_NAME, floor_extraction_dist, PARAM_FLOOR_EXTRACTION_DIST_DEFAULT);
        // Segmentation parameters, remove walls.
        add_param(PARAM_SEG_DISTANCE_NAME, seg_distance, PARAM_SEG_DISTANCE_DEFAULT);
        add_param(PARAM_SEG_PERCENTAGE_NAME, seg_percentage, PARAM_SEG_PERCENTAGE_DEFAULT);
        // Camera angle
        add_param(PARAM_CAM_ANGLE_NAME, cam_angle, PARAM_CAM_ANGLE_DEFAULT);
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

