## new docker run notes

### vehicle node setup

1. start docker container
    ```
    cd vehicle-cloud-inference
    ./docker/run.sh -a
    ```

2. switch to created 'rosuser' from default 'root' user
    - needed as 'rwthika/acdc:latest' docker image starts with root user and causes issues
```
su rosuser
# password: rosuser
```

3. setup ros workspace (first rime setup)
```
cd ..
./setup_new_ros_ws.sh
```
reference
```
rosuser@nemo:~/ws/catkin_workspace$ 
rosuser@nemo:~/ws/catkin_workspace$ cd ..
rosuser@nemo:~/ws$ ./setup_new_ros_ws.sh 
```
4. switch to workspace and source devel in shell
```
cd ws_mqtt_nodes
source devel/setup.bash
```

5. start node
```
roslaunch mqtt_node cloud_node.launch
```

#### expected terminal output // for reference
```

nemo@nemo:~/code/repos/vehicle-cloud-inference$ ./docker/run.sh -a
GPUs: all

Starting container ...
  Name: ros
  Image: rwthika/acdc:latest

a8bb4c7c8def5d1c1e8355e8d04b8ab19006828703ad6dba88226b9d865e7a12

=== ROS Docker Container =======================================================

Container setup:
- Ubuntu: 20.04.4 LTS (Focal Fossa) (user: rosuser, password: rosuser)
- Architecture: x86_64
- CUDA: 11.2.152
- cuDNN: 8.1.0
- TensorRT: 7.2.2
- Python: 3.8.10
- TensorFlow Python: 2.8.0
- TensorFlow C/C++: 2.8
- ROS: noetic
- CMake: 3.16.3

Available GPUs: 1
  name               driver_version   utilization.gpu [%]   utilization.memory [%]   memory.used [MiB]   memory.total [MiB]
  GeForce GTX 1050   455.32.00        0 %                   0 %                      154 MiB             4040 MiB

Template Commands: (https://gitlab.ika.rwth-aachen.de/automated-driving/docker#templates)
- Create new ROS package:            ros-add-package
  - Add node to package:               ros-add-node
  - Add nodelet to package:            ros-add-nodelet
- Initialize ROS GitLab repository:  ros-init-repo

rosuser@nemo:~/ws/catkin_workspace$ 

rosuser@nemo:~/ws/catkin_workspace$ cd ..

rosuser@nemo:~/ws$ su rosuser
Password: 
=== ROS Docker Container =======================================================

Container setup:
- Ubuntu: 20.04.4 LTS (Focal Fossa) (user: rosuser, password: rosuser)
- Architecture: x86_64
- CUDA: 11.2.152
- cuDNN: 8.1.0
- TensorRT: 7.2.2
- Python: 3.8.10
- TensorFlow Python: 2.8.0
- TensorFlow C/C++: 2.8
- ROS: noetic
- CMake: 3.16.3

Available GPUs: 1
  name               driver_version   utilization.gpu [%]   utilization.memory [%]   memory.used [MiB]   memory.total [MiB]
  GeForce GTX 1050   455.32.00        0 %                   0 %                      154 MiB             4040 MiB

Template Commands: (https://gitlab.ika.rwth-aachen.de/automated-driving/docker#templates)
- Create new ROS package:            ros-add-package
  - Add node to package:               ros-add-node
  - Add nodelet to package:            ros-add-nodelet
- Initialize ROS GitLab repository:  ros-init-repo

rosuser@nemo:~/ws$ ls
README.md  bag  catkin_workspace  docker  mqtt_node  setup_new_ros_ws.sh

rosuser@nemo:~/ws$ ./setup_new_ros_ws.sh 
preparing ros ws with package ...
create dir: ./ws_mqtt_nodes/src
copied mqtt_node to newly created workspace folder
moved to ros ws folder '/home/rosuser/ws/ws_mqtt_nodes'. running catkin_make ...
Base path: /home/rosuser/ws/ws_mqtt_nodes
Source space: /home/rosuser/ws/ws_mqtt_nodes/src
Build space: /home/rosuser/ws/ws_mqtt_nodes/build
Devel space: /home/rosuser/ws/ws_mqtt_nodes/devel
Install space: /home/rosuser/ws/ws_mqtt_nodes/install
Creating symlink "/home/rosuser/ws/ws_mqtt_nodes/src/CMakeLists.txt" pointing to "/opt/ros/noetic/share/catkin/cmake/toplevel.cmake"
####
#### Running command: "cmake /home/rosuser/ws/ws_mqtt_nodes/src -DCATKIN_DEVEL_PREFIX=/home/rosuser/ws/ws_mqtt_nodes/devel -DCMAKE_INSTALL_PREFIX=/home/rosuser/ws/ws_mqtt_nodes/install -G Unix Makefiles" in "/home/rosuser/ws/ws_mqtt_nodes/build"
####
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Using CATKIN_DEVEL_PREFIX: /home/rosuser/ws/ws_mqtt_nodes/devel
-- Using CMAKE_PREFIX_PATH: /opt/ros/noetic
-- This workspace overlays: /opt/ros/noetic
-- Found PythonInterp: /usr/bin/python3 (found suitable version "3.8.10", minimum required is "3") 
-- Using PYTHON_EXECUTABLE: /usr/bin/python3
-- Using Debian Python package layout
-- Found PY_em: /usr/lib/python3/dist-packages/em.py  
-- Using empy: /usr/lib/python3/dist-packages/em.py
-- Using CATKIN_ENABLE_TESTING: ON
-- Call enable_testing()
-- Using CATKIN_TEST_RESULTS_DIR: /home/rosuser/ws/ws_mqtt_nodes/build/test_results
-- Forcing gtest/gmock from source, though one was otherwise available.
-- Found gtest sources under '/usr/src/googletest': gtests will be built
-- Found gmock sources under '/usr/src/googletest': gmock will be built
-- Found PythonInterp: /usr/bin/python3 (found version "3.8.10") 
-- Found Threads: TRUE  
-- Using Python nosetests: /usr/bin/nosetests3
-- catkin 0.8.10
-- BUILD_SHARED_LIBS is on
-- BUILD_SHARED_LIBS is on
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- ~~  traversing 1 packages in topological order:
-- ~~  - mqtt_node
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- +++ processing catkin package: 'mqtt_node'
-- ==> add_subdirectory(mqtt_node)
-- Found OpenCV: /usr (found version "4.2.0") 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/rosuser/ws/ws_mqtt_nodes/build
####
#### Running command: "make -j8 -l8" in "/home/rosuser/ws/ws_mqtt_nodes/build"
####
prepared ros workspace with package.
rosuser@nemo:~/ws$ ls
README.md  bag  catkin_workspace  docker  mqtt_node  setup_new_ros_ws.sh  ws_mqtt_nodes
rosuser@nemo:~/ws$ cd ws_mqtt_nodes/
rosuser@nemo:~/ws/ws_mqtt_nodes$ . devel/setup.bash 
rosuser@nemo:~/ws/ws_mqtt_nodes$ 


```