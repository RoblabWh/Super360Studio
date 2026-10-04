# „Karte berechnen“ und RViz-Wiedergabe unter Ubuntu 24.04: ROS 2 Jazzy,
# Livox-SDK2, livox_ros_driver2 und FAST_LIO_ROS2 auf der App-Basis.
#
# Bauen (Kontext ist die Repo-Wurzel, gebraucht wird nur
# config/fastlio/whs_dense.yaml):
#
#   docker build -f docker/ubuntu-24.04/ros.Dockerfile -t super360-u2404-ros:latest .
#
# Die Workspaces ws_livox und fastlio2_ws liegen unter /opt/super360/ros und
# nicht im Home: HOME wechselt je nach Lauf (Testskripte setzen es auf den
# Ordner über den Bags). SUPER360_ROS_WS sagt der App, wo sie sind
# (core/ros_umgebung.py:workspace); ohne die Variable sucht sie wie auf dem
# Rechner unter ~. Die App sourct sie selbst und erkennt Jazzy dort auch.
#
# RViz zeichnet unter xvfb mit Mesa (llvmpipe, OpenGL 4.5); die NVIDIA-
# Fähigkeit "graphics" braucht es dafür nicht.
#
# Gegenüber 22.04 (Humble) gebaut aus denselben Ständen der Fremd-Repos wie
# auf dem Entwicklungsrechner (ARGs unten). Patches an Fremdcode stehen
# einzeln mit Grund an ihrem RUN.

ARG BASIS=super360-u2404-basis:latest
FROM ${BASIS}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Paketquelle von ROS 2 über das offizielle ros2-apt-source-Paket (bringt
# Schlüssel und sources-Eintrag mit und erneuert beide bei Schlüsselwechsel).
ARG ROS_APT_SOURCE_VERSION=1.3.0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && codename="$(. /etc/os-release && echo "$VERSION_CODENAME")" \
    && curl -fsSL -o /tmp/ros2-apt-source.deb \
       "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${codename}_all.deb" \
    && apt-get install -y /tmp/ros2-apt-source.deb \
    && rm -f /tmp/ros2-apt-source.deb \
    && rm -rf /var/lib/apt/lists/*

# ros-base enthält rosbag2 samt sqlite3-Speicher-Plugin (die Bags sind
# sqlite3, metadata version 5); das Plugin steht trotzdem ausdrücklich da,
# weil Jazzy mcap als Vorgabe hat. rviz2 für die Wiedergabe, mavros-msgs für
# GPSRAW/State (Nachrichtendefinitionen unter /opt/ros/jazzy/share).
# Der Rest sind die Bauabhängigkeiten von livox_ros_driver2 und FAST_LIO_ROS2
# (package.xml/CMakeLists.txt): pcl_ros, PCL, Eigen, APR, Python-Header
# (FAST-LIO sucht PythonLibs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-rosbag2-storage-default-plugins \
        ros-jazzy-rviz2 \
        ros-jazzy-mavros-msgs \
        ros-jazzy-pcl-ros ros-jazzy-pcl-conversions \
        ros-jazzy-rclcpp-components ros-jazzy-std-srvs \
        ros-jazzy-visualization-msgs ros-jazzy-common-interfaces \
        ros-jazzy-rosidl-default-generators ros-jazzy-ament-cmake-auto \
        python3-colcon-common-extensions \
        build-essential cmake git pkg-config \
        libpcl-dev libeigen3-dev libapr1-dev libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Stände wie auf dem 22.04-Rechner, auf dem die Referenzkarten entstanden.
ARG LIVOX_SDK2_REF=6a94015
ARG LIVOX_DRIVER_REF=6b9356c
ARG FASTLIO_REF=18418bc
ARG WS_HOME=/opt/super360/ros
ENV SUPER360_ROS_WS=$WS_HOME

# Livox-SDK2 nach /usr/local (dort sucht livox_ros_driver2 die Bibliothek).
# Patch: gcc 13 bindet <cstdint> nicht mehr indirekt über andere
# Standard-Header ein; mehrere SDK-Dateien nutzen uint8_t & Co. ohne eigenes
# #include und brechen sonst mit "'uint8_t' was not declared" ab. Statt jede
# Datei anzufassen, zieht -include cstdint den Header überall vor.
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /opt/Livox-SDK2 \
    && cd /opt/Livox-SDK2 && git checkout --quiet "$LIVOX_SDK2_REF" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CXX_FLAGS="-include cstdint" \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && ldconfig \
    && rm -rf /opt/Livox-SDK2/build

# livox_ros_driver2: nur der Nachrichtentyp CustomMsg der Mid-360 wird
# gebraucht, der Treiberknoten baut mit. Das build.sh des Repos kennt nur
# "humble"/"ROS2" und löscht den Workspace; die Schritte stehen hier direkt:
# package_ROS2.xml als package.xml, launch_ROS2 als launch, dann colcon mit
# den Schaltern, die build.sh setzen würde. HUMBLE_ROS=humble wählt in der
# CMakeLists.txt rosidl_get_typesupport_target, das es auch in Jazzy gibt;
# der andere Zweig nutzt interne Target-Namen der alten rosidl-Generatoren.
RUN mkdir -p "$WS_HOME/ws_livox/src" \
    && git clone https://github.com/Livox-SDK/livox_ros_driver2.git \
       "$WS_HOME/ws_livox/src/livox_ros_driver2" \
    && cd "$WS_HOME/ws_livox/src/livox_ros_driver2" \
    && git checkout --quiet "$LIVOX_DRIVER_REF" \
    && cp -f package_ROS2.xml package.xml \
    && cp -rf launch_ROS2/ launch/ \
    && cd "$WS_HOME/ws_livox" \
    && source /opt/ros/jazzy/setup.bash \
    && colcon build --cmake-args -DROS_EDITION=ROS2 -DHUMBLE_ROS=humble \
    && rm -rf build log

# FAST_LIO_ROS2 mit der Konfiguration aus diesem Repo (ohne --symlink-install
# wie auf 22.04: whs_dense.yaml wird beim Bauen nach install/ kopiert).
COPY config/fastlio/whs_dense.yaml /tmp/whs_dense.yaml
RUN mkdir -p "$WS_HOME/fastlio2_ws/src" \
    && git clone https://github.com/Ericsii/FAST_LIO_ROS2.git \
       "$WS_HOME/fastlio2_ws/src/FAST_LIO_ROS2" \
    && cd "$WS_HOME/fastlio2_ws/src/FAST_LIO_ROS2" \
    && git checkout --quiet "$FASTLIO_REF" \
    && git submodule update --init --recursive \
    && mv /tmp/whs_dense.yaml config/whs_dense.yaml \
    && cd "$WS_HOME/fastlio2_ws" \
    && source /opt/ros/jazzy/setup.bash \
    && source "$WS_HOME/ws_livox/install/setup.bash" \
    && colcon build \
    && rm -rf build log

# Läuft der Container mit --user, gehören die Workspaces root; lesen und
# sourcen genügt, geschrieben wird dort zur Laufzeit nichts.
RUN chmod -R a+rX "$WS_HOME/ws_livox" "$WS_HOME/fastlio2_ws"
