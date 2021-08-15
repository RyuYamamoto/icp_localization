#include <icp_localization/icp_localization.h>

int main(int argc, char** argv)
{
	ros::init(argc, argv, "icp_localization");

	ICPLocalization icp_localization;

	ros::spin();

	return 0;
}
