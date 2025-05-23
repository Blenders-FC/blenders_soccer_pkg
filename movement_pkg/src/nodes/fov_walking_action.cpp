/*
    Authors:
        Pedro Deniz
        Marlene Cobian
*/

#include "movement_pkg/nodes/fov_walking_action.h"


BT::FOVWalking::FOVWalking(std::string name) 
    : ActionNode::ActionNode(name), 
    WalkingController(), 
    FOVCalculation()
{
    type_ = BT::ACTION_NODE;
    thread_ = std::thread(&FOVWalking::WaitForTick, this);
}

BT::FOVWalking::~FOVWalking() {}

void BT::FOVWalking::WaitForTick()
{
    while(ros::ok())
    {
        ROS_TAGGED_ONCE_LOG("WAIT FOR TICK", "DEFAULT", false, "Wait_fov_walking");
        tick_engine.Wait();
        ROS_TAGGED_ONCE_LOG("TICK RECEIVED", "DEFAULT", false, "Received_fov_walking");

        // Flow for walking to ball - using FOV calculation
        
        while (get_status() == BT::IDLE)
        {
            set_status(BT::RUNNING);

            this->setModule("walking_module");
            ROS_COLORED_LOG("Walking towards target...", BLUE, false);
            walkTowardsTarget();

            if (walkingSucced)
            {
                ROS_SUCCESS_LOG("OP3 manager has reached the ball!");
                set_status(BT::SUCCESS);
            }
        }
    }
    ROS_ERROR_LOG("ROS stopped unexpectedly", false);
    set_status(BT::FAILURE);
}

void BT::FOVWalking::walkTowardsTarget()
{
    ros::Time curr_time_walk = ros::Time::now();
    ros::Duration dur_walk = curr_time_walk - prev_time_walk_;
    double delta_time_walk = dur_walk.nsec * 0.000000001 + dur_walk.sec;
    prev_time_walk_ = curr_time_walk;

    while (ros::ok())
    {
        
        distance_to_ball = ball->distance;
        pan_angle_to_ball = ball->pan_angle;

        if (distance_to_ball < 0) distance_to_ball *= (-1);
        ROS_COLORED_LOG("distance to ball: %f     pan angle: %f", PINK, false, distance_to_ball);
        
        if (distance_to_ball > distance_to_kick_)
        {
            fb_move = 0.0;
            rl_angle = 0.0;
            distance_to_walk = distance_to_ball - distance_to_kick_;

            calcFootstep(distance_to_walk, pan_angle_to_ball, delta_time_walk, fb_move, rl_angle);
            setWalkingParam(fb_move, 0, rl_angle, true);

            walk_command.data = "start";
            walk_command_pub.publish(walk_command);

            ros::Duration(0.1).sleep();
        }
        else{
            stopWalking();
            walkingSucced = true;
            break;
        }
    }
}

void BT::FOVWalking::Halt()
{
    stopWalking();

    set_status(BT::HALTED);
    ROS_TAGGED_ONCE_LOG("FOVWalking HALTED: Stopped fov walking", "ORANGE", false, "Halted_fov_walking");
}
