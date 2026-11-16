#!/usr/bin/env bash
# Read-only inventory. Run in the existing ROS terminal; nothing is installed.
# Usage: PX4_DIR="$HOME/PX4-Autopilot" bash tools/inspect_environment.sh > environment.txt
set -u

probe() {
    printf '\n[%s]\n' "$1"
    shift
    if command -v "$1" >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            timeout 15s "$@" 2>&1 || printf 'Command unavailable, failed, or timed out (status %s).\n' "$?"
        else
            "$@" 2>&1 || printf 'Command failed (status %s).\n' "$?"
        fi
    else
        printf 'Not available in this terminal: %s\n' "$1"
    fi
}

printf 'Heterogeneous swarm host inventory; read-only; UTC '
date -u '+%Y-%m-%dT%H:%M:%SZ'
probe 'OS release' lsb_release -a
probe 'Kernel and architecture' uname -srmo
probe 'ROS distribution' rosversion -d
probe 'roscpp' rosversion roscpp
probe 'rospy' rosversion rospy
probe 'MAVROS' rosversion mavros
probe 'Gazebo Classic' gazebo --version
probe 'New Gazebo CLI, if installed' gz --versions
probe 'System Python' /usr/bin/python3 --version
probe 'Python 3.11, if on PATH' python3.11 --version
probe 'Git' git --version
probe 'CPU' lscpu
probe 'Memory' free -h

printf '\n[Selected ROS shell variables]\n'
printf 'ROS_DISTRO=%s\nROS_VERSION=%s\n' "${ROS_DISTRO:-unset}" "${ROS_VERSION:-unset}"
printf 'Other environment variables and credentials are deliberately not collected.\n'

if [ -n "${PX4_DIR:-}" ] && [ -d "$PX4_DIR" ]; then
    probe 'PX4 revision' git -C "$PX4_DIR" rev-parse HEAD
    probe 'PX4 tag/description' git -C "$PX4_DIR" describe --tags --always
    probe 'PX4 working tree changes (filenames/status only)' git -C "$PX4_DIR" status --short
    probe 'PX4 submodule revisions' git -C "$PX4_DIR" submodule status --recursive
else
    printf '\n[PX4]\nSet PX4_DIR to the existing source checkout and run again.\n'
fi

printf '\nNo ROS nodes, Gazebo server, PX4 process, install, upgrade, or simulation was started.\n'
