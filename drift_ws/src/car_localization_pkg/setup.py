"""Setuptools configuration for the car_localization_pkg ROS 2 Python package."""

from glob import glob
import os

from setuptools import find_packages, setup

package_name = "car_localization_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="Source-agnostic car localization: canonical car frames and state.",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "mount_tf_node = car_localization_pkg.mount_tf_node:main",
            "car_state_node = car_localization_pkg.car_state_node:main",
        ],
    },
)
