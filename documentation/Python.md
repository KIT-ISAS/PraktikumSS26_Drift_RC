# Python

### Venv and ROS
In order to install new packages using pip, we need a virtual environment (venv) as it is restricted by Ubuntu to install packages globally. The venv is located at `/opt/venv`. To activate the venv, run the following command:

```bash
source /opt/venv/bin/activate
```
Now all packages installed using pip will be installed into `/opt/venv/lib/python3.12/site-packages`. To make these packages available for ROS, we need to add this path to the `PYTHONPATH` environment variable, which is done in the Dockerfile. However, this is not really a "good practice" as it can lead to conflicts between packages installed in the venv and packages installed globally. The venv packages have lower precedence because they are added to the **end** of the `PYTHONPATH`. This way, ROS packages will not be conflicted. **It may cause troubles with your custom packages though.**