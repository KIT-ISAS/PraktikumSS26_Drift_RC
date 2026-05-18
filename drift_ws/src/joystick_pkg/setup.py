from setuptools import find_packages, setup

package_name = "joystick_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/joystick_reader.launch.py",
                "launch/joystick_mapper.launch.py",
                "launch/joy.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/joystick_params.yaml",
                "config/joystick_mapper_params.yaml",
                "config/joy_params.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="louis.wagner@ka-raceing.de",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "joystick_reader = joystick_pkg.joystick_reader_node:main",
            "joystick_mapper = joystick_pkg.joystick_mapper_node:main",
        ],
    },
)
