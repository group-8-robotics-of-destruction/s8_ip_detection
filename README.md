s8_ip_detection
===============

This node takes the cloud from the primesense and filters out the object that we wan't to find.

It does it in the following way:

1. Use a passthrough filter to reduce the field of view.
2. Use a voxel grid to downsample the cloud to make it faster.
3. Remove walls and floor using planar detection in pcl.
4. rotate pointcloud so that it's aligned with the floor.
5. Use another passthrough filter to remove along y direction after rotation.
6. Cluster the cloud to isolate remaning objects and only allow those of sizes similar to  the objects.
7. Publish the most likely cluster (Might have to improve).

The output cluster is published to /s8/detectedObjects, the point cloud is of type
pcl::PointXYZRGB