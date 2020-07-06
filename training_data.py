################################################################################
# Copyright (C) 2012-2013 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################
import sys
import os, inspect

#src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
#src_dir = "/mnt/c/Users/sahil/Coding/Python/SummerHacks"
#arch_dir = '../lib/x64' if sys.maxsize > 2**32 else '../lib/x86'
#sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))

import Leap, thread, time

class SampleListener(Leap.Listener):
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    bone_names = ['metacarpal', 'proximal', 'intermediate', 'distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def on_init(self, controller):
        # determine what file number to write to
        file_to_write = "peace"
        count = 102

        # get all the files within the data directory
        basedir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(basedir, 'csv_data/')
        files = os.listdir(data_dir)
        file_hash = {}

        # turn the file data into a hash for O(1)
        for i in range(0, len(files)):
            file_hash[files[i]] = 0

        # determine what count number this is
        test_string = file_to_write + str(count) + ".csv"
        while True:
            if test_string in file_hash:
                count += 1
                test_string = file_to_write + str(count) + ".csv"
            else:
                self.file_count = count
                break

        # create a file there
        data_file = os.path.join(data_dir, test_string)
        self.f = open(data_file, "w")
        self.f.write("frame_id,timestamp,")
        self.f.write("hand_position_x,hand_position_y,hand_position_z,")
        self.f.write("pitch,roll,yaw,arm_x,arm_y,arm_z,wrist_x,wrist_y,wrist_z,")
        self.f.write("elbow_x,elbow_y,elbow_z,")

        for finger in self.finger_names:
            for bone in self.bone_names:
                self.f.write(finger + "_" + bone + "_" + "start_x,")
                self.f.write(finger + "_" + bone + "_" + "start_y,")
                self.f.write(finger + "_" + bone + "_" + "start_z,")
                self.f.write(finger + "_" + bone + "_" + "end_x,")
                self.f.write(finger + "_" + bone + "_" + "end_y,")
                self.f.write(finger + "_" + bone + "_" + "end_z,")
                self.f.write(finger + "_" + bone + "_" + "direction_x,")
                self.f.write(finger + "_" + bone + "_" + "direction_y,")
                self.f.write(finger + "_" + bone + "_" + "direction_z,")

        print("Initialized")

    def on_connect(self, controller):
        print("Connected")
        print("Click enter once to start recording. Click enter twice to stop.")
        sys.stdin.readline()
        print("Recording: " + str(self.file_count))

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
