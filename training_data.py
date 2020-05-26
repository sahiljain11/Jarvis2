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
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        print("Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
              frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures())))

        # Get hands
        for hand in frame.hands:

            handType = "Left hand" if hand.is_left else "Right hand"

            # only read in one hand
            if handType == "Left Hand":
                return
            # else continue

            # write the palm coordinates
            self.f.write("\n" + "{:.4f}".format(hand.palm_position[0]) + "," + "{:.4f}".format(hand.palm_position[1]) + "," + "{:.4f}".format(hand.palm_position[2]) + ",")

            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction

            # record the pitch, roll, yaw
            self.f.write("{:.4f}".format(direction.pitch) + "," + "{:.4f}".format(normal.roll) + "," + "{:.4f}".format(direction.yaw) + ",")

            # arm positions
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
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        listener.f.close()
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
