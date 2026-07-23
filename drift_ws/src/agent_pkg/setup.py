from glob import glob
import os

from setuptools import find_packages, setup

package_name = "agent_pkg"

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
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Louis Wagner",
    maintainer_email="louis.wagner0610@gmail.com",
    description="ROS2 wrapper around the trained RL drift agent",
    license="Apache License 2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "agent_node = agent_pkg.agent_node:main",
            "mpc_node = agent_pkg.mpc_node:main",
            "data_collection_node = agent_pkg.data_collection_node:main",
            "demo_recorder_node = agent_pkg.demo_recorder_node:main",
        ],
    },
)
