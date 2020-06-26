from flask import request
import json
import sys
import os, inspect
import numpy as np

import Leap, thread, time

class SampleListener(Leap.Listener):
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    bone_names = ['metacarpal', 'proximal', 'intermediate', 'distal']

    def on_init(self, controller):
        self.json_data = []
        self.initial_x = [0] * len(self.finger_names)
        self.prev_gesture = 0
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

    def find_distance_between_vec(x1, y1, z1, x2, y2, z2):
        return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        # TODO: Test hands (if 0, reset)
        
        # Get hands
        for hand in frame.hands:

            handType = "Left hand" if hand.is_left else "Right hand"

            # only read in one hand
            if handType == "Left Hand":
                self.initial_x = [0] * len(self.finger_names)
                return
            # else continue

            raw_data = {}

            # get hand positions x,y,z
            raw_data["hand_x"] = hand.palm_position[0]
            raw_data["hand_y"] = hand.palm_position[1]
            raw_data["hand_z"] = hand.palm_position[2]

            # get wrist positions x,y,z
            raw_data["wrist_x"] = hand.arm.wrist_position[0]
            raw_data["wrist_y"] = hand.arm.wrist_position[1]
            raw_data["wrist_z"] = hand.arm.wrist_position[2]

            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction

            # Get fingers
            j = 0
            for finger in hand.fingers:
                for b in range(0, 4):
                    bone = finger.bone(b)

                    # x, y, z
                    # prev_joint == "start"
                    raw_data[self.finger_names[j] + "_" + bone_names[b] + "start_x"] = bone.prev_joint[0]
                    raw_data[self.finger_names[j] + "_" + bone_names[b] + "start_y"] = bone.prev_joint[1]
                    raw_data[self.finger_names[j] + "_" + bone_names[b] + "start_z"] = bone.prev_joint[2]
                        
                    # next_joint == "end"
                    raw_data[self.finger_names[j] + "_" + bone_names[b] + "end_x"] = bone.next_joint[0]
                    raw_data[self.finger_names[j] + "_" + bone_names[b] + "end_y"] = bone.next_joint[1]
                    raw_data[self.finger_names[j] + "_" + bone_names[b] + "end_z"] = bone.next_joint[2]

                j += 1

            model_data = {}

            # distance betwene thumb distals and the other fingers' distals
            for i in range(1, len(self.finger_names)):
                model_data["thumb2" + self.finger_names[i]] = find_distance_between_vec(raw_data["thumb_distal_end_x"],
                                                                                        raw_data["thumb_distal_end_y"],
                                                                                        raw_data["thumb_distal_end_z"],
                                                                                        raw_data[self.finger_names[i] + "_distal_end_x"],
                                                                                        raw_data[self.finger_names[i] + "_distal_end_y"],
                                                                                        raw_data[self.finger_names[i] + "_distal_end_z"])

            # TODO: add initial distance x
            # change in each finger location
            for i in range(0, len(self.finger_names)):
                if len(self.json_data) == 0:
                    model_data[self.finger_names[i] + "fromstart"] = 0
                else:
                    model_data[self.finger_names[i] + "fromstart"] = raw_data[self.finger_names[i] + "_distal_end_x"] - self.initial_x[i]


            # length between the hand to the end of the finger
            for i in range(0, len(self.finger_names)):
                model_data[self.finger_names[i] + "Length"] = find_distance_between_vec(raw_data["hand_x"], raw_data["hand_y"], raw_data["hand_z"],
                                                                                        raw_data[self.finger_names[i] + "_distal_end_x"],
                                                                                        raw_data[self.finger_names[i] + "_distal_end_y"],
                                                                                        raw_data[self.finger_names[i] + "_distal_end_z"])

            # calculate angles omega
            for i in range(1, len(self.finger_names)):
                temp_col_x = raw_data[self.finger_names[i] + "_intermediate_end_x"] - raw_data[self.finger_names[i] + "_intermediate_start_x"]
                temp_col_y = raw_data[self.finger_names[i] + "_intermediate_end_y"] - raw_data[self.finger_names[i] + "_intermediate_start_y"]
                temp_col_z = raw_data[self.finger_names[i] + "_intermediate_end_z"] - raw_data[self.finger_names[i] + "_intermediate_start_z"]

                temp_col_a = raw_data[self.finger_names[i] + "_distal_end_x"] - raw_data[self.finger_names[i] + "_distal_start_x"]
                temp_col_b = raw_data[self.finger_names[i] + "_distal_end_y"] - raw_data[self.finger_names[i] + "_distal_start_y"]
                temp_col_c = raw_data[self.finger_names[i] + "_distal_end_z"] - raw_data[self.finger_names[i] + "_distal_start_z"]

                temp_col = (temp_col_x * temp_col_a) + (temp_col_y * temp_col_b) + (temp_col_z * temp_col_c)
                temp_col = temp_col / (find_distance_between_vec(raw_data[self.finger_names[i] + "_intermediate_end_x"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_end_y"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_end_z"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_start_x"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_start_y"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_start_z"]))

                temp_col = temp_col / (find_distance_between_vec(raw_data[self.finger_names[i] + "_distal_end_x"],
                                                                 raw_data[self.finger_names[i] + "_distal_end_y"],
                                                                 raw_data[self.finger_names[i] + "_distal_end_z"],
                                                                 raw_data[self.finger_names[i] + "_distal_start_x"],
                                                                 raw_data[self.finger_names[i] + "_distal_start_y"],
                                                                 raw_data[self.finger_names[i] + "_distal_start_z"]))

                model_data[self.finger_names[i] + "_omega"] = np.arccos(temp_col)

            ## calculate angles beta
            for i in range(1, len(self.finger_names)):
                temp_col_x = raw_data[self.finger_names[i] + "_proximal_end_x"] - raw_data[self.finger_names[i] + "_proximal_start_x"]
                temp_col_y = raw_data[self.finger_names[i] + "_proximal_end_y"] - raw_data[self.finger_names[i] + "_proximal_start_y"]
                temp_col_z = raw_data[self.finger_names[i] + "_proximal_end_z"] - raw_data[self.finger_names[i] + "_proximal_start_z"]

                temp_col_a = raw_data[self.finger_names[i] + "_intermediate_end_x"] - raw_data[self.finger_names[i] + "_intermediate_start_x"]
                temp_col_b = raw_data[self.finger_names[i] + "_intermediate_end_y"] - raw_data[self.finger_names[i] + "_intermediate_start_y"]
                temp_col_c = raw_data[self.finger_names[i] + "_intermediate_end_z"] - raw_data[self.finger_names[i] + "_intermediate_start_z"]

                temp_col = (temp_col_x * temp_col_a) + (temp_col_y * temp_col_b) + (temp_col_z * temp_col_c)
                temp_col = temp_col / (find_distance_between_vec(raw_data[self.finger_names[i] + "_proximal_end_x"],
                                                                 raw_data[self.finger_names[i] + "_proximal_end_y"],
                                                                 raw_data[self.finger_names[i] + "_proximal_end_z"],
                                                                 raw_data[self.finger_names[i] + "_proximal_start_x"],
                                                                 raw_data[self.finger_names[i] + "_proximal_start_y"],
                                                                 raw_data[self.finger_names[i] + "_proximal_start_z"]))

                temp_col = temp_col / (find_distance_between_vec(raw_data[self.finger_names[i] + "_intermediate_end_x"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_end_y"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_end_z"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_start_x"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_start_y"],
                                                                 raw_data[self.finger_names[i] + "_intermediate_start_z"]))

                model_data[self.finger_names[i] + "_beta"] = np.arccos(temp_col)

            # calculate angles gamma
            for i in range(1, len(self.finger_names)):
                temp_col_x = raw_data[self.finger_names[i] + "_proximal_end_x"] - raw_data[self.finger_names[i] + "_proximal_start_x"]
                temp_col_y = raw_data[self.finger_names[i] + "_proximal_end_y"] - raw_data[self.finger_names[i] + "_proximal_start_y"]
                temp_col_z = raw_data[self.finger_names[i] + "_proximal_end_z"] - raw_data[self.finger_names[i] + "_proximal_start_z"]

                temp_col_a = raw_data[self.finger_names[i - 1] + "_proximal_end_x"] - raw_data[self.finger_names[i - 1] + "_proximal_start_x"]
                temp_col_b = raw_data[self.finger_names[i - 1] + "_proximal_end_y"] - raw_data[self.finger_names[i - 1] + "_proximal_start_y"]
                temp_col_c = raw_data[self.finger_names[i - 1] + "_proximal_end_z"] - raw_data[self.finger_names[i - 1] + "_proximal_start_z"]

                temp_col = (temp_col_x * temp_col_a) + (temp_col_y * temp_col_b) + (temp_col_z * temp_col_c)

                temp_col = temp_col / (find_distance_between_vec(raw_data[self.finger_names[i] + "_proximal_end_x"],
                                                                 raw_data[self.finger_names[i] + "_proximal_end_y"],
                                                                 raw_data[self.finger_names[i] + "_proximal_end_z"],
                                                                 raw_data[self.finger_names[i] + "_proximal_start_x"],
                                                                 raw_data[self.finger_names[i] + "_proximal_start_y"],
                                                                 raw_data[self.finger_names[i] + "_proximal_start_z"]))

                temp_col = temp_col / (find_distance_between_vec(raw_data[self.finger_names[i - 1] + "_proximal_end_x"],
                                                                 raw_data[self.finger_names[i - 1] + "_proximal_end_y"],
                                                                 raw_data[self.finger_names[i - 1] + "_proximal_end_z"],
                                                                 raw_data[self.finger_names[i - 1] + "_proximal_start_x"],
                                                                 raw_data[self.finger_names[i - 1] + "_proximal_start_y"],
                                                                 raw_data[self.finger_names[i - 1] + "_proximal_start_z"]))

                model_data[self.finger_names[i - 1] + "_" + self.finger_names[i] + "_gamma"] = np.arccos(temp_col)

            # calculate the flick of the wrists
            wrist_to_palm_x = raw_data["hand_x"] - raw_data["wrist_x"]
            wrist_to_palm_y = raw_data["hand_y"] - raw_data["wrist_y"]
            wrist_to_palm_z = raw_data["hand_z"] - raw_data["wrist_z"]
            raw_data["wrist_phi"] = np.arctan(wrist_to_palm_y / wrist_to_palm_x)
            raw_data["wrist_theta"] = np.arctan((wrist_to_palm_x**2 + wrist_to_palm_y**2)**0.5 / wrist_to_palm_z)

            # update initial x
            if len(self.json_data) == 0:
                for i in range(0, len(self.finger_names)):
                    self.initial_x[i] = [raw_data[self.finger_names[i] + "_distal_end_x"]]

            # maintain 20 timesteps
            if len(self.json_data) < 20:
                self.json_data.append(raw_data)
            else:
                del self.json_data[0]
                self.json_data.append(raw_data)

            # send request if enough data
            if len(self.json_data) < 20:
                send = json.dumps(self.json_data)
                res = requests.post("http://127.0.0.1:3141/determine_gesture/", json=send).json()
                curr_gesture = res['gesture']

                # gets rid of chains of gestures
                if self.prev_gesture != 0:
                    self.prev_gesture = curr_gesture
                    curr_gesture = 0

                # computed gesture. send to QTC GUI
            


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
