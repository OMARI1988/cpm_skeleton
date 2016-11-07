#!/usr/bin/env python

import rospy
from strands_executive_msgs import task_utils
from strands_executive_msgs.msg import Task
from strands_executive_msgs.srv import DemandTask, SetExecutionStatus
# import strands_executive_msgs
import sys

def get_services():
    # get services necessary to do the jon
    demand_task_srv_name = '/task_executor/demand_task'
  #  set_exe_stat_srv_name = '/task_executor/set_execution_status'
    rospy.loginfo("Waiting for task_executor service...")
    rospy.wait_for_service(demand_task_srv_name)
   # rospy.wait_for_service(set_exe_stat_srv_name)
    rospy.loginfo("cpm Done")
    add_tasks_srv = rospy.ServiceProxy(demand_task_srv_name, DemandTask)
#    set_execution_status = rospy.ServiceProxy(set_exe_stat_srv_name, SetExecutionStatus)
    return add_tasks_srv#, set_execution_status


if __name__ == '__main__':
    rospy.init_node("demand_cpm_task")

    if len(sys.argv)==2:
        duration=rospy.Duration(int(sys.argv[1]))
        print "Duration: ", duration
    else:
        print "Usage: demand_activity_recognition_task.py waypoint_name duration (secs)"
        sys.exit(1)

    # Set the task exec
    demand_task = get_services()
    task= Task(action='cpm_action', max_duration=rospy.Duration(28800), start_node_id=waypoint)
    task_utils.add_duration_argument(task, duration)
    task_utils.add_string_argument(task, waypoint)

    resp = demand_task(task)
    print 'demanded task as id: %s' % resp.task_id
    rospy.loginfo('Success: %s' % resp.success)
    rospy.loginfo('Wait: %s' % resp.remaining_execution_time)
