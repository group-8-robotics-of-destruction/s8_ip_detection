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
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
// OTHER
#include <vector>


// DEFINITIONS
#define HZ                  10
#define BUFFER_SIZE         10

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

typedef pcl::PointXYZ PointT;
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
		passthroughFilterCloud(cloud);
		voxelGridCloud(cloud);
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		extractFloor(cloud, coefficients);
		if (coefficients->values.size() < 2)
			return;
		double theta = getAngle(coefficients);
		cout << "theta: " << theta << endl;

		rotatePointCloud(cloud, cloud, theta);
		Eigen::Vector3f mass_center;
		getMoments(cloud, &mass_center);
		cout << "x: " << mass_center(0) << " y: " << mass_center(1) << " z: " << mass_center(2) << endl;

		cloudPublish(cloud);
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

	void getMoments(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, Eigen::Vector3f *mass_center)
	{
		pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
		feature_extractor.setInputCloud (inputCloud);
		feature_extractor.compute ();
		feature_extractor.getMassCenter (*mass_center);
	}

	void rotatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud, double theta)
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
	void passthroughFilterCloud(pcl::PointCloud<PointT>::Ptr cloud_filter)
    {
		pcl::PassThrough<PointT> pass;

		pass.setInputCloud (cloud_filter);
		pass.setFilterFieldName ("z");
		pass.setFilterLimits (0.3, filter_z);
		pass.filter (*cloud_filter);

		pass.setInputCloud (cloud_filter);
		pass.setFilterFieldName ("x");
		pass.setFilterLimits (-filter_x, filter_x);
		pass.filter (*cloud_filter);

		pass.setInputCloud (cloud_filter);
		pass.setFilterFieldName ("y");
		pass.setFilterLimits (-filter_y, 0.4);
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

