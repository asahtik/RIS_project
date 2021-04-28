# HW4 - how to run a demo:

notes: you have to have installed ***dlib*** python module `python 2.7 -m pip install dlib`

Run the commands in seperate terminals. Wait for each to fully load.
```
roscore

roslaunch hw4 rins_world.launch

roslaunch hw4 amcl_simulation.launch

roslaunch hw4 view_navigation.launch

roslaunch hw4 localize_faces.launch
```