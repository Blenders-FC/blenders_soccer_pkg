/*
    Authors:
        Pedro Deniz
        Marlene Cobian
*/

#ifndef FOV_WALKING_ACTION_H
#define FOV_WALKING_ACTION_H

#include "movement_pkg/walking_controller.h"
#include "movement_pkg/fov_calculation.h"
#include <action_node.h>


namespace BT
{
class FOVWalking : public ActionNode, public WalkingController, public FOVCalculation
{
    public:
        // Constructor
        explicit FOVWalking(std::string name);
        ~FOVWalking();

        // The method that is going to be executed by the thread
        void WaitForTick();

        // The method used to interrupt the execution of the node
        void Halt();

    private:
        //  Auxiliar methods
        void walkTowardsTarget();

        // Variables
        double fb_move;
        double rl_angle;
        double distance_to_walk;
        const double distance_to_kick_ = 0.30;  // 0.22
        bool walkingSucced = false;
        std_msgs::String walk_command;
        ros::Time prev_time_walk_ = ros::Time::now();

        const TargetInfo* ball = blackboard.getTarget("ball");
        double pan_angle_to_ball;
        double distance_to_ball;
};
}  // namespace BT

#endif  // FOV_WALKING_ACTION_H
