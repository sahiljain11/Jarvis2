################################################################################
# Copyright (C) 2012-2013 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################

import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture

class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def on_init(self, controller):
        self.f = open("nothing.csv", "w")
        self.f.write("frame_id,timestamp,hand_position_x,hand_position_y,hand_position_z,pitch,roll,yaw,arm_x,arm_y,arm_z,wrist_x,wrist_y,wrist_z,elbow_x,elbow_y,elbow_z,thumb_metacarpal_start_x,thumb_metacarpal_start_y,thumb_metacarpal_start_z,thumb_metacarpal_end_x,thumb_metacarpal_end_y,thumb_metacarpal_end_z,thumb_metacarpal_direction_x,thumb_metacarpal_direction_y,thumb_metacarpal_direction_z,thumb_proximal_start_x,thumb_proximal_start_y,thumb_proximal_start_z,thumb_proximal_end_x,thumb_proximal_end_y,thumb_proximal_end_z,thumb_proximal_direction_x,thumb_proximal_direction_y,thumb_proximal_direction_z,thumb_intermediate_start_x,thumb_intermediate_start_y,thumb_intermediate_start_z,thumb_intermediate_end_x,thumb_intermediate_end_y,thumb_intermediate_end_z,thumb_intermediate_direction_x,thumb_intermediate_direction_y,thumb_intermediate_direction_z,thumb_distal_start_x,thumb_distal_start_y,thumb_distal_start_z,thumb_distal_end_x,thumb_distal_end_y,thumb_distal_end_z,thumb_distal_direction_x,thumb_distal_direction_y,thumb_distal_direction_z,index_metacarpal_start_x,index_metacarpal_start_y,index_metacarpal_start_z,index_metacarpal_end_x,index_metacarpal_end_y,index_metacarpal_end_z,index_metacarpal_direction_x,index_metacarpal_direction_y,index_metacarpal_direction_z,index_proximal_start_x,index_proximal_start_y,index_proximal_start_z,index_proximal_end_x,index_proximal_end_y,index_proximal_end_z,index_proximal_direction_x,index_proximal_direction_y,index_proximal_direction_z,index_intermediate_start_x,index_intermediate_start_y,index_intermediate_start_z,index_intermediate_end_x,index_intermediate_end_y,index_intermediate_end_z,index_intermediate_direction_x,index_intermediate_direction_y,index_intermediate_direction_z,index_distal_start_x,index_distal_start_y,index_distal_start_z,index_distal_end_x,index_distal_end_y,index_distal_end_z,index_distal_direction_x,index_distal_direction_y,index_distal_direction_z,middle_metacarpal_start_x,middle_metacarpal_start_y,middle_metacarpal_start_z,middle_metacarpal_end_x,middle_metacarpal_end_y,middle_metacarpal_end_z,middle_metacarpal_direction_x,middle_metacarpal_direction_y,middle_metacarpal_direction_z,middle_proximal_start_x,middle_proximal_start_y,middle_proximal_start_z,middle_proximal_end_x,middle_proximal_end_y,middle_proximal_end_z,middle_proximal_direction_x,middle_proximal_direction_y,middle_proximal_direction_z,middle_intermediate_start_x,middle_intermediate_start_y,middle_intermediate_start_z,middle_intermediate_end_x,middle_intermediate_end_y,middle_intermediate_end_z,middle_intermediate_direction_x,middle_intermediate_direction_y,middle_intermediate_direction_z,middle_distal_start_x,middle_distal_start_y,middle_distal_start_z,middle_distal_end_x,middle_distal_end_y,middle_distal_end_z,middle_distal_direction_x,middle_distal_direction_y,middle_distal_direction_z,ring_metacarpal_start_x,ring_metacarpal_start_y,ring_metacarpal_start_z,ring_metacarpal_end_x,ring_metacarpal_end_y,ring_metacarpal_end_z,ring_metacarpal_direction_x,ring_metacarpal_direction_y,ring_metacarpal_direction_z,ring_proximal_start_x,ring_proximal_start_y,ring_proximal_start_z,ring_proximal_end_x,ring_proximal_end_y,ring_proximal_end_z,ring_proximal_direction_x,ring_proximal_direction_y,ring_proximal_direction_z,ring_intermediate_start_x,ring_intermediate_start_y,ring_intermediate_start_z,ring_intermediate_end_x,ring_intermediate_end_y,ring_intermediate_end_z,ring_intermediate_direction_x,ring_intermediate_direction_y,ring_intermediate_direction_z,ring_distal_start_x,ring_distal_start_y,ring_distal_start_z,ring_distal_end_x,ring_distal_end_y,ring_distal_end_z,ring_distal_direction_x,ring_distal_direction_y,ring_distal_direction_z,pinky_metacarpal_start_x,pinky_metacarpal_start_y,pinky_metacarpal_start_z,pinky_metacarpal_end_x,pinky_metacarpal_end_y,pinky_metacarpal_end_z,pinky_metacarpal_direction_x,pinky_metacarpal_direction_y,pinky_metacarpal_direction_z,pinky_proximal_start_x,pinky_proximal_start_y,pinky_proximal_start_z,pinky_proximal_end_x,pinky_proximal_end_y,pinky_proximal_end_z,pinky_proximal_direction_x,pinky_proximal_direction_y,pinky_proximal_direction_z,pinky_intermediate_start_x,pinky_intermediate_start_y,pinky_intermediate_start_z,pinky_intermediate_end_x,pinky_intermediate_end_y,pinky_intermediate_end_z,pinky_intermediate_direction_x,pinky_intermediate_direction_y,pinky_intermediate_direction_z,pinky_distal_start_x,pinky_distal_start_y,pinky_distal_start_z,pinky_distal_end_x,pinky_distal_end_y,pinky_distal_end_z,pinky_distal_direction_x,pinky_distal_direction_y,pinky_distal_direction_z")
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")
        print("Click enter once to start recording. Click enter twice to stop.")
        sys.stdin.readline()
        print("Recording")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        #print(str(self.record))

        #print("Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures())))

        # Get hands
        for hand in frame.hands:

            handType = "Left hand" if hand.is_left else "Right hand"

            # only read in one hand
            if handType == "Left Hand":
                return
            # else continue

            # frame ID and timestamp
            self.f.write("\n" + str(frame.id) + "," + str(frame.timestamp) + ",")

            # write the palm coordinates
            self.f.write("{:.4f}".format(hand.palm_position[0]) + "," + "{:.4f}".format(hand.palm_position[1]) + "," + "{:.4f}".format(hand.palm_position[2]) + ",")

            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction

            # record the pitch, roll, yaw
            self.f.write("{:.4f}".format(direction.pitch) + "," + "{:.4f}".format(normal.roll) + "," + "{:.4f}".format(direction.yaw) + ",")

            # arm positions
            arm = hand.arm
            self.f.write("{:.4f}".format(arm.direction[0]) + "," + "{:.4f}".format(arm.direction[1]) + "," + "{:.4f}".format(arm.direction[2]) + ",")

            # wrist positions
            self.f.write("{:.4f}".format(arm.wrist_position[0]) + "," + "{:.4f}".format(arm.wrist_position[1]) + "," + "{:.4f}".format(arm.wrist_position[2]) + ",")
            
            # elbow positions
            self.f.write("{:.4f}".format(arm.elbow_position[0]) + "," + "{:.4f}".format(arm.elbow_position[1]) + "," + "{:.4f}".format(arm.elbow_position[2]) + ",")

            # Get fingers
            for finger in hand.fingers:
                for b in range(0, 4):
                    bone = finger.bone(b)

                    # write to file the coordinates for the components of the finger
                    self.f.write("{:.4f}".format(bone.prev_joint[0]) + "," + "{:.4f}".format(bone.prev_joint[1]) + "," + "{:.4f}".format(bone.prev_joint[2]) + ",")
                    self.f.write("{:.4f}".format(bone.next_joint[0]) + "," + "{:.4f}".format(bone.prev_joint[1]) + "," + "{:.4f}".format(bone.next_joint[2]) + ",")
                    self.f.write("{:.4f}".format(bone.direction[0])  + "," + "{:.4f}".format(bone.direction[1])  + "," + "{:.4f}".format(bone.direction[2])  + ",")


    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"

def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
        sys.stdin.readline()
    except KeyboardInterrupt:
        listener.f.close()
        pass
    finally:
        # Remove the sample listener when done
        listener.f.close()
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
