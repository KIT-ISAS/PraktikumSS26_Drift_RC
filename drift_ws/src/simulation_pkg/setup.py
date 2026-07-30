from glob import glob
import os

from setuptools import find_packages, setup

package_name = "simulation_pkg"

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
    maintainer="Julian Bihl",
    maintainer_email="julian-bihl@t-online.de",
    description="ROS2 wrapper for drift_core models",
    license="Apache License 2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "simulation_node = simulation_pkg.simulation_node:main",
        ],
    },
)
