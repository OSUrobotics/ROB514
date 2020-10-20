#!/usr/bin/env python3

# Get the windowing packages
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize, QPoint

from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor

import numpy as np

from world_state import WorldState
from door_sensor import DoorSensor
from robot_state import RobotState
from robot_state_estimation import RobotStateEstimation


# A helper class that implements a slider with given start and end float value; displays values
class SliderFloatDisplay(QWidget):
    gui = None

    def __init__(self, name, low, high, initial_value, ticks=100):
        """
        Give me a name, the low and high values, and an initial value to set
        :param name: Name displayed on slider
        :param low: Minimum value slider returns
        :param high: Maximum value slider returns
        :param initial_value: Should be a value between low and high
        :param ticks: Resolution of slider - all sliders are integer/fixed number of ticks
        """
        # Save input values
        self.name = name
        self.low = low
        self.range = high - low
        self.ticks = ticks

        # I'm a widget with a text value next to a slider
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(ticks)
        # call back - call change_value when slider changed
        self.slider.valueChanged.connect(self.change_value)

        self.display = QLabel()
        self.set_value(initial_value)
        self.change_value()

        layout.addWidget(self.display)
        layout.addWidget(self.slider)

    # Use this to get the value between low/high
    def value(self):
        """Return the current value of the slider"""
        return (self.slider.value() / self.ticks) * self.range + self.low

    # Called when the value changes - resets text
    def change_value(self):
        if SliderFloatDisplay.gui is not None:
            SliderFloatDisplay.gui.repaint()
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))

    # Use this to change the value (does clamping)
    def set_value(self, value_f):
        value_tick = self.ticks * (value_f - self.low) / self.range
        value_tick = min(max(0, value_tick), self.ticks)
        self.slider.setValue(int(value_tick))
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))


# A helper class that implements a slider with given start and end value; displays values
class SliderIntDisplay(QWidget):
    gui = None

    def __init__(self, name, min_ticks, max_ticks, initial_value):
        """
        Give me a name, the low and high values, and an initial value to set
        :param name: Name displayed on slider
        :param min_ticks: Minimum value slider returns
        :param max_ticks: Maximum value slider returns
        :param initial_value: Should be a value between low and high
        """
        # Save input values
        self.name = name
        self.min_ticks = min_ticks
        self.max_ticks = max_ticks

        # I'm a widget with a text value next to a slider
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_ticks)
        self.slider.setMaximum(max_ticks)
        # call back - call change_value when slider changed
        self.slider.valueChanged.connect(self.change_value)

        self.display = QLabel()
        self.set_value(initial_value)
        self.change_value()

        layout.addWidget(self.display)
        layout.addWidget(self.slider)

    # Use this to get the value between low/high
    def value(self):
        """Return the current value of the slider"""
        return self.slider.value()

    # Called when the value changes - resets text
    def change_value(self):
        if SliderIntDisplay.gui is not None:
            SliderIntDisplay.gui.repaint()
        self.display.setText('{}: {}'.format(self.name, self.value()))

    # Use this to change the value (does clamping)
    def set_value(self, value_tick):
        value_tick = min(max(self.min_ticks, value_tick), self.max_ticks)
        self.slider.setValue(int(value_tick))
        self.change_value()


# The main class for handling the robot drawing and geometry
class DrawRobotAndWalls(QWidget):
    def __init__(self, gui_world):
        super().__init__()

        # In order to get to the slider values
        self.gui = gui_world

        self.title = "Robot and Walls"

        # Window size
        self.top = 15
        self.left = 15
        self.width = 700
        self.height = 700

        # State/action text
        self.sensor_text = "No sensor"
        self.action_text = "No action"
        self.move_text = "No move"
        self.loc_text = "No location"

        # The world state
        self.world_state = WorldState()

        # For querying door
        self.door_sensor = DoorSensor()

        # For moving robot
        self.robot_state = RobotState()

        # For robot state estimation
        self.state_estimation = RobotStateEstimation()

        # For keeping sampled error
        self.last_wall_sensor_noise = 0
        self.last_move_noise = 0

        # Height of prob
        self.draw_height = 0.6

        # Set geometry
        self.text = "None"
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

    # What to draw
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_robot(qp)
        self.draw_wall(qp)
        if self.gui.draw_kalman:
            self.draw_robot_gauss(qp)
            self.draw_move_gauss(qp)
        else:
            self.draw_probabilities(qp)

        self.draw_wall_gauss(qp)
        self.draw_world(qp)
        self.draw_sensor_action(qp, event)
        qp.end()

    # Put some text in the bottom left
    def draw_sensor_action(self, qp, _):
        qp.setPen(QColor(168,34,3))
        qp.setFont(QFont('Decorative', 10))
        #  Put sensor text in lower left...
        text_loc = QPoint(self.x_map(0.01), self.y_map(self.draw_height))
        qp.drawText(text_loc, self.sensor_text)
        # .. action text in lower right
        text_loc = QPoint(self.x_map(0.7), self.y_map(self.draw_height))
        qp.drawText(text_loc, self.action_text)
        # .. loc text in upper ;eft
        text_loc = QPoint(self.x_map(0.01), self.y_map(self.draw_height + 0.025))
        qp.drawText(text_loc, self.loc_text)
        # .. move text in upper right
        text_loc = QPoint(self.x_map(0.7), self.y_map(self.draw_height + 0.025))
        qp.drawText(text_loc, self.move_text)

    def draw_wall(self, qp):
        pen = QPen(Qt.black, 4, Qt.SolidLine)
        brush = QBrush(Qt.CrossPattern)
        qp.setPen(pen)
        qp.setBrush(brush)

        qp.drawRect(self.x_map(0), self.y_map(1.0), self.x_map(1), self.in_pixels(0.2))

        pen.setColor(Qt.red)
        brush.setStyle(Qt.SolidPattern)
        qp.setPen(pen)
        qp.setBrush(brush)

        door_width = self.world_state.door_width
        for d in self.world_state.doors:
            qp.drawRect(self.x_map( d - door_width/2 ), self.y_map(0.95), self.in_pixels(door_width), self.in_pixels(0.15))

    def draw_probabilities(self, qp):
        pen = QPen(Qt.black, 1, Qt.SolidLine)
        qp.setPen(pen)

        div = 1 / len(self.state_estimation.probabilities)
        for i in range(1, len(self.state_estimation.probabilities)):
            qp.drawLine(self.x_map(i*div), self.y_map(0.0), self.x_map(i*div), self.y_map(self.draw_height*0.8))

        pen.setColor(Qt.blue)
        qp.setPen(pen)
        for i, p in enumerate(self.state_estimation.probabilities):
            qp.drawLine(self.x_map(i * div), self.y_map(self.draw_height * p), self.x_map((i + 1) * div), self.y_map(self.draw_height * p))

    def draw_world(self, qp):
        pen = QPen(Qt.black, 3, Qt.SolidLine)
        qp.setPen(pen)

        qp.drawLine(self.x_map(0), self.y_map(0.0), self.x_map(1), self.y_map(0.0))
        qp.drawLine(self.x_map(0), self.y_map(0.0), self.x_map(0), self.y_map(self.draw_height))
        qp.drawLine(self.x_map(1), self.y_map(0.0), self.x_map(1), self.y_map(self.draw_height))

    def draw_robot_gauss(self, qp):
        pen = QPen(Qt.darkBlue, 1, Qt.DotLine)
        qp.setPen(pen)

        dx = np.linspace(0, 1, 200)
        mu = self.state_estimation.mean
        standard_deviation = self.state_estimation.standard_deviation
        dy = [np.exp(-np.power(mu - x, 2.0) / (2 * np.power(standard_deviation, 2.0))) for x in dx]
        pts = []
        # Protect against sd set to zero/NaN
        max_y = max(np.exp(-np.power(0, 2.0) / (2 * np.power(standard_deviation, 2.0))), 1e-12)
        if max_y < 1e-6:
            max_y = 1e-6
        for x, y in zip(dx, dy):
            pts.append(QPoint(self.x_map(x), self.y_map(0.5 * y*self.draw_height/max_y)))
        for i in range(0, len(pts)-1):
            qp.drawLine(pts[i], pts[i+1])

    # Wall sensor distribution
    def draw_wall_gauss(self, qp):
        pen = QPen(Qt.gray, 1, Qt.DashLine)
        qp.setPen(pen)

        dx = np.linspace(0, 1, 200)
        mu = self.robot_state.robot_loc
        standard_deviation = self.world_state.wall_standard_deviation
        dy = [np.exp(-np.power(mu - x, 2.0) / (2 * np.power(standard_deviation, 2.0))) for x in dx]
        pts = []
        # Protect against sd set to zero/NaN
        max_y = max(np.exp(-np.power(0, 2.0) / (2 * np.power(standard_deviation, 2.0))), 1e-12)
        height = 0.1 / max_y
        for x, y in zip(dx, dy):
            pts.append(QPoint(self.x_map(x), self.y_map(self.draw_height + 0.05 + y*height)))
        for i in range(0, len(pts)-1):
            qp.drawLine(pts[i], pts[i+1])

        # Put a dashed line indicating last noise sample
        pen.setColor(Qt.red)
        pen.setWidth(1)
        qp.setPen(pen)
        qp.drawLine(QPoint(self.x_map(self.robot_state.robot_loc + self.last_wall_sensor_noise), self.y_map(self.draw_height + 0.1)),
                    QPoint(self.x_map(self.robot_state.robot_loc + self.last_wall_sensor_noise), self.y_map(self.draw_height + 0.045)))

    # Wall sensor distribution
    def draw_move_gauss(self, qp):
        pen = QPen(Qt.gray, 1, Qt.DashLine)
        qp.setPen(pen)

        dx = np.linspace(0, 1, 200)
        mu = self.robot_state.robot_loc
        standard_deviation = self.robot_state.robot_move_standard_deviation_err
        dy = [np.exp(-np.power(mu - x, 2.0) / (2 * np.power(standard_deviation, 2.0))) for x in dx]
        pts = []
        # Protect against sd set to zero/NaN
        max_y = max(np.exp(-np.power(0, 2.0) / (2 * np.power(standard_deviation, 2.0))), 1e-12)
        height = 0.2
        for x, y in zip(dx, dy):
            pts.append(QPoint(self.x_map(x), self.y_map(y*height/max_y)))
        for i in range(0, len(pts)-1):
            qp.drawLine(pts[i], pts[i+1])

        # Put a dashed line indicating the last noise sample for move
        pen.setColor(Qt.red)
        pen.setWidth(1.5)
        qp.setPen(pen)
        qp.drawLine(QPoint(self.x_map(self.robot_state.robot_loc + self.last_move_noise), self.y_map(0)),
                    QPoint(self.x_map(self.robot_state.robot_loc + self.last_move_noise), self.y_map(0.075)))

    def draw_robot(self, qp):
        pen = QPen(Qt.darkMagenta, 2, Qt.SolidLine)
        qp.setPen(pen)

        x_i = self.x_map(self.robot_state.robot_loc)
        y_i = self.y_map(0.09)
        qp.drawLine(x_i-5, y_i, x_i+5, y_i)
        qp.drawLine(x_i, y_i-5, x_i, y_i+5)

    # Map from [0,1]x[0,1] to the width and height of the window
    def x_map(self, x):
        return int(x * self.width)

    # Map from [0,1]x[0,1] to the width and height of the window - need to flip y
    def y_map(self, y):
        return self.height - int(y * self.height) - 1

    def in_pixels(self, v):
        return int(v * self.height)

    def query_door_sensor(self):
        front_door_yn = self.door_sensor.is_in_front_of_door(self.world_state, self.robot_state)
        sensor_value = self.door_sensor.sensor_reading(self.world_state, self.robot_state)
        self.sensor_text = "Sensor reading: {}, actual: {}".format(sensor_value, front_door_yn)
        return sensor_value


class StateEstimationGUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('ROB 456 State Estimation')

        # Set this to true when you start homework 3
        # begin homework 3 : Problem 0
        self.draw_kalman = False
        # end homework 3 : Problem 0

        # Control buttons for the interface
        left_side_layout = self._init_left_layout_()
        middle_layout = self._init_middle_layout_()
        right_side_layout = self._init_right_layout_()

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        # Three side-by-side panes
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)
        top_level_layout.addLayout(right_side_layout)

        # Make the variables match the sliders
        self.robot_scene.state_estimation.reset_probabilities(self.n_probabilities.value())
        self.robot_scene.robot_state.adjust_location(self.n_probabilities.value())
        self.set_probabilities()
        self.set_gauss()

        # So the sliders can update robot_scene
        SliderIntDisplay.gui = self
        SliderFloatDisplay.gui = self

    # Set up the left set of sliders/buttons (state estimation)
    def _init_left_layout_(self):
        # Two reset buttons one for probabilities, one for doors
        reset_probabilities_button = QPushButton('Reset probabilities')
        reset_probabilities_button.clicked.connect(self.reset_probabilities)

        reset_random_doors_button = QPushButton('Random doors')
        reset_random_doors_button.clicked.connect(self.random_doors)

        resets = QGroupBox('Resets')
        resets_layout = QVBoxLayout()
        resets_layout.addWidget(reset_probabilities_button)
        resets_layout.addWidget(reset_random_doors_button)
        resets.setLayout(resets_layout)

        # Action/sensor buttons - update state estimation
        query_wall_sensor_button = QPushButton('Query wall sensor')
        query_wall_sensor_button.clicked.connect(self.query_wall_sensor)

        query_door_sensor_button = QPushButton('Query door sensor')
        query_door_sensor_button.clicked.connect(self.query_door_sensor)

        move_left_button = QPushButton('Move left')
        move_left_button.clicked.connect(self.move_left)

        move_right_button = QPushButton('Move right')
        move_right_button.clicked.connect(self.move_right)

        s_and_a = QGroupBox('State estimation: query state and do action')
        s_and_a_layout = QVBoxLayout()
        s_and_a_layout.addWidget(query_wall_sensor_button)
        s_and_a_layout.addWidget(query_door_sensor_button)
        s_and_a_layout.addWidget(move_left_button)
        s_and_a_layout.addWidget(move_right_button)
        s_and_a.setLayout(s_and_a_layout)

        # The parameters of the robot we're simulating (world/door/robot state)
        parameters = QGroupBox('World parameters, use update button to set')
        parameter_layout = QVBoxLayout()
        self.n_doors = SliderIntDisplay('Number doors', 1, 6, 3)
        self.n_probabilities = SliderIntDisplay('Number probabilities', 10, 200, 20)
        self.prob_see_door_if_door = SliderFloatDisplay('Prob see door if door', 0.01, 0.99, 0.8)
        self.prob_see_door_if_not_door = SliderFloatDisplay('Prob see door if not door', 0.01, 0.99, 0.1)
        self.prob_move_left_if_left = SliderFloatDisplay('Prob move left if left', 0.1, 0.85, 0.8)
        self.prob_move_right_if_left = SliderFloatDisplay('Prob move right if left', 0.0, 0.1, 0.05)
        self.prob_move_right_if_right = SliderFloatDisplay('Prob move right if right', 0.1, 0.85, 0.8)
        self.prob_move_left_if_right = SliderFloatDisplay('Prob move left if right', 0.0, 0.1, 0.05)
        update_world_state = QPushButton('Update world state')
        update_world_state.clicked.connect(self.set_probabilities)

        parameter_layout.addWidget(self.n_doors)
        parameter_layout.addWidget(self.n_probabilities)
        parameter_layout.addWidget(self.prob_see_door_if_door)
        parameter_layout.addWidget(self.prob_see_door_if_not_door)
        parameter_layout.addWidget(self.prob_move_left_if_left)
        parameter_layout.addWidget(self.prob_move_right_if_left)
        parameter_layout.addWidget(self.prob_move_right_if_right)
        parameter_layout.addWidget(self.prob_move_left_if_right)
        parameter_layout.addWidget(update_world_state)

        parameters.setLayout(parameter_layout)

        # Put all the pieces in one box
        left_side_layout = QVBoxLayout()

        left_side_layout.addWidget(resets)
        left_side_layout.addStretch()
        left_side_layout.addWidget(s_and_a)
        left_side_layout.addWidget(parameters)

        return left_side_layout

    # Drawing screen and quit button
    def _init_middle_layout_(self):
        # The display for the robot drawing
        self.robot_scene = DrawRobotAndWalls(self)

        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget(self.robot_scene)
        mid_layout.addWidget(quit_button)

        return mid_layout

    # Right side sliders/buttons (Gaussian/Kalman filtering)
    def _init_right_layout_(self):
        # The means of the robot we're simulating (Gaussian)
        parameters_gauss = QGroupBox('Gauss parameters, use update button to set')
        parameter_layout_gauss = QVBoxLayout()
        self.prob_see_wall_SD = SliderFloatDisplay('Prob wall SD', 0.001, 0.3, 0.01)
        self.prob_move_SD = SliderFloatDisplay('Prob move SD', 0.001, 0.01, 0.005)
        update_world_state_gauss = QPushButton('Update gauss')
        update_world_state_gauss.clicked.connect(self.set_gauss)

        parameter_layout_gauss.addWidget(self.prob_see_wall_SD)
        parameter_layout_gauss.addWidget(self.prob_move_SD)
        parameter_layout_gauss.addWidget(update_world_state_gauss)

        parameters_gauss.setLayout(parameter_layout_gauss)

        # Update buttons for Kalman filer
        reset_kalman_button = QPushButton('Reset Kalman')
        reset_kalman_button.clicked.connect(self.reset_kalman)

        query_wall_sensor_button_gauss = QPushButton('Query wall/distance sensor')
        query_wall_sensor_button_gauss.clicked.connect(self.query_wall_sensor_button_kalman)

        self.move_gauss_amount = SliderFloatDisplay('Amount move', -0.1, 0.1, 0.0)

        move_button_gauss = QPushButton('Move')
        move_button_gauss.clicked.connect(self.move_kalman)

        s_and_a_gauss = QGroupBox('Prediction/correction steps (Kalman)')
        s_and_a_layout_gauss = QVBoxLayout()
        s_and_a_layout_gauss.addWidget(reset_kalman_button)
        s_and_a_layout_gauss.addWidget(query_wall_sensor_button_gauss)
        s_and_a_layout_gauss.addWidget(move_button_gauss)
        s_and_a_layout_gauss.addWidget(self.move_gauss_amount)
        s_and_a_gauss.setLayout(s_and_a_layout_gauss)

        # stack them all up
        right_side_layout = QVBoxLayout()
        right_side_layout.addWidget(parameters_gauss)
        right_side_layout.addStretch()
        right_side_layout.addWidget(s_and_a_gauss)

        return right_side_layout

    # set robot back in middle
    def reset_probabilities(self):
        self.robot_scene.state_estimation.reset_probabilities(self.n_probabilities.value())
        self.robot_scene.robot_state.adjust_location(self.n_probabilities.value())
        self.robot_scene.repaint()

    def random_doors(self):
        self.robot_scene.world_state.random_door_placement(self.n_doors.value(), self.n_probabilities.value())
        self.robot_scene.repaint()

    def set_probabilities(self):
        self.reset_probabilities()
        self.robot_scene.world_state.random_door_placement(self.n_doors.value(), self.n_probabilities.value())
        self.robot_scene.door_sensor.set_probabilities(self.prob_see_door_if_door.value(),
                                                       self.prob_see_door_if_not_door.value())
        self.robot_scene.robot_state.set_move_left_probabilities(self.prob_move_left_if_left.value(),
                                                                 self.prob_move_right_if_left.value())
        self.robot_scene.robot_state.set_move_right_probabilities(self.prob_move_right_if_right.value(),
                                                                  self.prob_move_left_if_right.value())
        self.repaint()

    def set_gauss(self):
        self.robot_scene.robot_state.set_move_gauss_probabilities(self.prob_move_SD.value())
        self.robot_scene.world_state.set_wall_standard_deviation(self.prob_see_wall_SD.value())
        self.repaint()

    def query_wall_sensor(self):
        dist_wall_z = self.robot_scene.world_state.query_wall(self.robot_scene.robot_state)
        dist_wall_actual = self.robot_scene.robot_state.robot_loc
        self.robot_scene.last_wall_sensor_noise = dist_wall_z - dist_wall_actual
        self.robot_scene.loc_text = "Asked loc {0:0.2f}, got {1:0.2f}".format(dist_wall_actual, dist_wall_z)
        self.robot_scene.state_estimation.update_dist_sensor(self.robot_scene.world_state, dist_wall_z)

        self.draw_kalman = False
        self.repaint()

    def query_door_sensor(self):
        returned_sensor_reading = self.robot_scene.query_door_sensor()
        self.robot_scene.state_estimation.update_belief_sensor_reading(self.robot_scene.world_state,
                                                                       self.robot_scene.door_sensor,
                                                                       returned_sensor_reading)
        self.draw_kalman = False
        self.repaint()

    def move_left(self):
        div = 1/self.n_probabilities.value()
        step = self.robot_scene.robot_state.move_left(div)
        if step < 0:
            self.robot_scene.action_text = "Asked move left, moved left"
        elif step > 0:
            self.robot_scene.action_text = "Asked move left, moved right"
        else:
            self.robot_scene.action_text = "Asked move left, did not move"

        self.robot_scene.state_estimation.update_belief_move_left(self.robot_scene.robot_state)

        self.draw_kalman = False
        self.repaint()

    def move_right(self):
        div = 1/self.n_probabilities.value()
        step = self.robot_scene.robot_state.move_right(div)
        if step > 0:
            self.robot_scene.action_text = "Asked move right, moved right"
        elif step < 0:
            self.robot_scene.action_text = "Asked move right, moved left"
        else:
            self.robot_scene.action_text = "Asked move right, did not move"

        self.robot_scene.state_estimation.update_belief_move_right(self.robot_scene.robot_state)

        self.draw_kalman = False
        self.repaint()

    def reset_kalman(self):
        self.robot_scene.state_estimation.reset_kalman()
        self.robot_scene.last_wall_reading = self.robot_scene.robot_state.robot_loc
        self.robot_scene.last_move_request = self.robot_scene.robot_state.robot_loc

        self.draw_kalman = True
        self.repaint()

    def query_wall_sensor_button_kalman(self):
        dist_wall_z = self.robot_scene.world_state.query_wall(self.robot_scene.robot_state)
        dist_wall_actual = self.robot_scene.robot_state.robot_loc
        self.robot_scene.last_wall_sensor_noise = dist_wall_actual - dist_wall_z
        self.robot_scene.loc_text = "Asked loc {0:0.2f}, got {1:0.2f}".format(dist_wall_actual, dist_wall_z)
        self.robot_scene.state_estimation.update_gauss_sensor_reading(self.robot_scene.world_state, dist_wall_z)

        self.draw_kalman = True
        self.repaint()

    def move_kalman(self):
        amount_requested = self.move_gauss_amount.value()
        amount = self.robot_scene.robot_state.move_gauss(amount_requested)
        self.robot_scene.last_move_noise = amount - amount_requested
        self.robot_scene.move_text = "Asked move {0:0.4f}, moved {1:0.4f}".format(amount_requested, amount)
        self.robot_scene.state_estimation.update_kalman_move(self.robot_scene.robot_state,
                                                             self.move_gauss_amount.value())

        self.draw_kalman = True
        self.repaint()

    def draw(self, _):
        self.robot_scene.draw()


if __name__ == '__main__':
    app = QApplication([])

    gui = StateEstimationGUI()

    gui.show()

    app.exec_()
