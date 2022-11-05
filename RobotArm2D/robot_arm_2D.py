#!/usr/bin/env python3

# Get the windowing packages
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize

from PyQt5.QtGui import QPainter, QPen, QFont, QColor

from random import random

import numpy as np
from gui_sliders import SliderFloatDisplay
import arm_forward_kinematics as afk
import arm_ik_gradient as ik_gradient
import arm_ik_jacobian as ik_jacobian


# The main class for handling the robot drawing and geometry
class DrawRobot(QWidget):
    def __init__(self, in_gui):
        super().__init__()

        # In order to get to the slider values
        self.gui = in_gui

        # Title of the window
        self.title = "Robot arm"
        # output text displayed in window
        self.text = "Not reaching"

        # Window size
        self.top = 15
        self.left = 15
        self.width = 500
        self.height = 500

        # The geometry from the arm_forward_kinematics.py file
        self.arm_angles, self.arm_with_angles = self.build_arm_and_angles()

        # Set size of window
        self.init_ui()

    def init_ui(self):
        self.text = "Not reaching"
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # For making sure the window shows up the right size
    def minimumSizeHint(self):
        return QSize(self.width, self.height)

    # For making sure the window shows up the right size
    def sizeHint(self):
        return QSize(self.width, self.height)

    # What to draw - called whenever window needs to be drawn
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_text(event, qp)
        self.draw_target(qp)
        self.draw_arm(qp)
        qp.end()

    # Put some text in the bottom left
    def draw_text(self, event, qp):
        qp.setPen(QColor(168, 84, 3))
        qp.setFont(QFont('Decorative', 16))
        qp.drawText(event.rect(), Qt.AlignBottom, self.text)

    # Map from [-0.5,1]x[0.5,1] to the width and height of the window
    def x_map(self, x):
        return int((x+0.5) * self.width)

    # Map from [0,1]x[0,1] to the width and height of the window - need to flip y
    def y_map(self, y):
        return self.height - int(y * self.height) - 1

    # Draw a + where the target is and another where the gripper grasp is
    def draw_target(self, qp):
        pen = QPen(Qt.darkRed, 2, Qt.SolidLine)
        qp.setPen(pen)
        x_i = self.x_map(self.gui.reach_x.value())
        y_i = self.y_map(self.gui.reach_y.value())
        qp.drawLine(x_i-5, y_i, x_i+5, y_i)
        qp.drawLine(x_i, y_i-5, x_i, y_i+5)

    def draw_arm_end_pt(self, x, y, vx, vy, qp):
        pen = QPen(Qt.darkGreen, 2, Qt.SolidLine)
        qp.setPen(pen)

        x_i = self.x_map(x)
        y_i = self.y_map(y)
        qp.drawLine(x_i-5, y_i, x_i+5, y_i)
        qp.drawLine(x_i, y_i-5, x_i, y_i+5)
        vx_i = self.x_map(x + 0.2 * vx)
        vy_i = self.y_map(y + 0.2 * vy)
        qp.drawLine(x_i, y_i, vx_i, vy_i)

    # Draw a line on the screen
    def draw_line(self, qp, x, y, xnext, ynext):
        """ Map (0,1) X (0,1) to (0,H) X (W,0) """
        qp.drawLine(self.x_map(x), self.y_map(y), self.x_map(xnext), self.y_map(ynext))

    # Draw the given box
    def draw_obj(self, obj, in_matrix, qp):
        """ Draw an obj - the same as object_in-world plot_object_in_world_coord_system"""
        pen_color = Qt.black
        if obj["Color"] == "darkturquoise":
            pen_color = Qt.darkCyan
        elif obj["Color"] == "darkgoldenrod":
            pen_color = Qt.darkYellow
        elif obj["Color"] == "darkgreen":
            pen_color = Qt.darkGreen
        pen = QPen(pen_color, 2, Qt.SolidLine)
        qp.setPen(pen)

        matrix = in_matrix
        if in_matrix is None:
            matrix = np.identity(3)

        # This multiplies the matrix by the points
        pts_in_world = matrix @ obj["Matrix"] @ obj["Pts"]

        for i in range(0, pts_in_world.shape[1]-1):
            i_next = i+1
            self.draw_line(qp, pts_in_world[0, i], pts_in_world[1, i], pts_in_world[0, i_next], pts_in_world[1, i_next])

    def build_arm_and_angles(self):
        """ Build the arm from the current sliders (lengths and angles).
        This is computationally wasteful, but it gets around the problem of the arm and sliders being out of sync
        @return Angles and arm geometry"""
        base_size_param = (0.25, 0.125)
        link_sizes_param = []
        for l in self.gui.length_slds[0:3]:
            link_sizes_param.append((l.value(), 0.25 * l.value()))
        palm_width_param = self.gui.length_slds[3].value()
        finger_size_param = (self.gui.length_slds[4].value(), 0.25 * self.gui.length_slds[4].value())

        # This function calls each of the set_transform_xxx functions, and puts the results
        # in a list (the gripper - the last element - is a list)
        arm_geometry = afk.create_arm_geometry(base_size_param, link_sizes_param, palm_width_param, finger_size_param)

        angles_for_arm = []
        # Get the angles for the links
        for l in self.gui.theta_slds[0:-3]:
            angles_for_arm.append(l.value())

        # Now the angles for the gripper
        angles_for_arm.append([self.gui.theta_slds[-3].value(), self.gui.theta_slds[-2].value(), self.gui.theta_slds[-1].value()])
        afk.set_angles_of_arm_geometry(arm_geometry, angles_for_arm)
        return angles_for_arm, arm_geometry

    def set_slider_values_from_angles(self, angles_for_arm):
        """Set all the sliders from the angles
        @param angles_for_arm - the link angles plus the wrist/finger"""
        # Get the angles for the links
        for a, l in zip(angles_for_arm[:-1], self.gui.theta_slds[0:-3]):
            l.set_value(a)  # Set the slider to the angle

        # Now the angles for the gripper
        gripper_angles = angles_for_arm[-1]
        # Wrist
        self.gui.theta_slds[-3].set_value(gripper_angles[0])
        # Fingers
        self.gui.theta_slds[-2].set_value(gripper_angles[1])
        self.gui.theta_slds[-1].set_value(gripper_angles[2])

    def draw_arm(self, qp):
        """ Get the current angles from the sliders then set the matrices then draw
        @param: qp - the painter window
        """

        self.arm_angles, self.arm_with_angles = self.build_arm_and_angles()
        matrices = afk.get_matrices_all_links(self.arm_with_angles)

        # Now draw - essentially arm_forward_kinematics plot_complete_arm
        for i, component in enumerate(self.arm_with_angles[:-1]):
            self.draw_obj(component, matrices[i] @ afk.get_rotation_link(component), qp)

        gripper = self.arm_with_angles[-1]

        # The palm
        wrist_rotation = afk.get_rotation_link(gripper[0])
        self.draw_obj(gripper[0], matrices[-1] @ wrist_rotation, qp)

        for finger in gripper[1:3]:
            self.draw_obj(finger, matrices[-1] @ wrist_rotation @ afk.get_matrix_finger(finger), qp)

        # Draw the gripper grasp point
        x, y = afk.get_gripper_location(self.arm_with_angles)

        # ... and orientation
        vx, vy = afk.get_gripper_orientation(self.arm_with_angles)
        self.draw_arm_end_pt(x, y, vx, vy, qp)


class RobotArmGUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('ROB 514 2D robot arm')

        # Control buttons for the interface
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Different do reach commands
        reach_gradient_button = QPushButton('Reach gradient')
        reach_gradient_button.clicked.connect(self.reach_gradient)

        reach_jacobian_button = QPushButton('Reach Jacobian')
        reach_jacobian_button.clicked.connect(self.reach_jacobian)

        reaches = QGroupBox('Reaches')
        reaches_layout = QHBoxLayout()
        reaches_layout.addWidget(reach_gradient_button)
        reaches_layout.addWidget(reach_jacobian_button)
        reaches.setLayout(reaches_layout)

        # The parameters of the robot arm we're simulating
        parameters_len = QGroupBox('Arm lengths')
        parameter_len_layout = QVBoxLayout()

        parameters_ang = QGroupBox('Arm angles')
        parameter_ang_layout = QVBoxLayout()

        self.theta_slds = []
        self.length_slds = []
        component_names = ["Link 0", "Link 1", "Link 2", "Wrist", "Finger 1", "Finger 2"]
        for n in component_names:
            self.theta_slds.append(SliderFloatDisplay('Angle ' + n, -np.pi/2, np.pi/2, 0))
            parameter_ang_layout.addWidget(self.theta_slds[-1])

        sldr_bds = (0.1, 0.3, 0.2)
        scl_bds = 1.0
        for n in component_names[0:-2]:
            self.length_slds.append(SliderFloatDisplay('Length ' + n, sldr_bds[0] * scl_bds, sldr_bds[1] * scl_bds, sldr_bds[2] * scl_bds))
            parameter_len_layout.addWidget(self.length_slds[-1])
            scl_bds *= 0.8

        self.length_slds.append(SliderFloatDisplay('Length fingers', sldr_bds[0] * scl_bds, sldr_bds[1] * scl_bds, sldr_bds[2] * scl_bds))
        parameter_len_layout.addWidget(self.length_slds[-1])

        parameters_ang.setLayout(parameter_ang_layout)
        parameters_len.setLayout(parameter_len_layout)

        # The point to reach to
        reach_point = QGroupBox('Reach point')
        reach_point_layout = QHBoxLayout()
        self.reach_x = SliderFloatDisplay('x', -0.5, 0.5, 0.5)
        self.reach_y = SliderFloatDisplay('y', 0, 1, 0.5)
        random_button = QPushButton('Random')
        random_button.clicked.connect(self.random_reach)
        reach_point_layout.addWidget(self.reach_x)
        reach_point_layout.addWidget(self.reach_y)
        reach_point_layout.addWidget(random_button)
        reach_point.setLayout(reach_point_layout)

        # The display for the graph
        self.robot_arm = DrawRobot(self)

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)
        left_side_layout = QVBoxLayout()
        right_side_layout = QVBoxLayout()

        left_side_layout.addWidget(reaches)
        left_side_layout.addWidget(reach_point)
        left_side_layout.addStretch()

        parameters_both = QGroupBox('Arm parameters')
        parameters_both_layout = QHBoxLayout()
        parameters_both.setLayout(parameters_both_layout)
        parameters_both_layout.addWidget(parameters_ang)
        parameters_both_layout.addWidget(parameters_len)
        left_side_layout.addWidget(parameters_both)

        right_side_layout.addWidget(self.robot_arm)
        right_side_layout.addWidget(quit_button)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(right_side_layout)

        SliderFloatDisplay.gui = self

    def update_simulation_parameters(self):
        """ Redraw with angles, lengths"""
        self.robot_arm.repaint()

    # generate a random reach point
    def random_reach(self):
        self.reach_x.set_value(random())
        self.reach_y.set_value(random())
        self.robot_arm.repaint()

    def reach_gradient(self):
        """Align the robot end point (palm) to the target point using gradient descent"""
        # Where the target is (from the slider)
        target = np.array([self.reach_x.value(), self.reach_y.value()])
        # Do one step, and update the angles

        angles, arm_geometry = self.robot_arm.build_arm_and_angles()
        old_dist = ik_gradient.distance_to_goal(arm_geometry, np.transpose(target))
        b_succ, new_angles, count = ik_gradient.gradient_descent(arm_geometry, angles, np.transpose(target), True)
        if b_succ:
            afk.set_angles_of_arm_geometry(arm_geometry, new_angles)
        else:
            afk.set_angles_of_arm_geometry(arm_geometry, angles)
        new_dist = ik_gradient.distance_to_goal(arm_geometry, np.transpose(target))

        if not b_succ:
            self.robot_arm.text = f"Arm did not move\ndist {old_dist:0.3}\ncount {count}"
        else:
            self.robot_arm.text = f"Arm moved\nold {old_dist:0.3} new {new_dist:0.3}\ncount {count}"
            self.robot_arm.set_slider_values_from_angles(new_angles)
        self.robot_arm.repaint()

    def reach_jacobian(self):
        """ Use the Jacobian to change the angles"""
        # Where the target is (from the slider)
        target = np.array([self.reach_x.value(), self.reach_y.value()])
        # Do one step, and update the angles

        angles, arm_geometry = self.robot_arm.build_arm_and_angles()
        old_dist = ik_gradient.distance_to_goal(arm_geometry, np.transpose(target))
        b_succ, new_angles, count = ik_jacobian.jacobian_follow_path(arm_geometry, angles, np.transpose(target), True)
        if b_succ:
            afk.set_angles_of_arm_geometry(arm_geometry, new_angles)
        else:
            afk.set_angles_of_arm_geometry(arm_geometry, angles)
        new_dist = ik_gradient.distance_to_goal(arm_geometry, np.transpose(target))

        if np.all(np.isclose(new_angles[0:-1], angles[0:-1])) and np.isclose(new_angles[-1][0], angles[-1][0]):
            self.robot_arm.text = f"Arm did not move\ndist {old_dist:0.3}\ncount {count}"
            self.robot_arm.set_slider_values_from_angles(angles)
        else:
            self.robot_arm.text = f"Arm moved\nold {old_dist:0.3} new {new_dist:0.3}\ncount {count}"
            self.robot_arm.set_slider_values_from_angles(new_angles)
        self.robot_arm.repaint()

    def draw(self, unused_data):
        self.robot_arm.draw_arm()


if __name__ == '__main__':
    app = QApplication([])

    gui = RobotArmGUI()

    gui.show()

    app.exec_()
