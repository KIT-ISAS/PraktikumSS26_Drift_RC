# Black Formatter

Black is the Python formatter used for this project. It can be run with the command `black .` from the root of the repository. This will format all Python files in the repository according to the Black style guide. It is also part of the CI Pipeline, so any code that does not conform to the Black style will cause the CI checks to fail.

### VSCode Integration
If you are using VSCode, you can set up Black as your default formatter. To do this, you must install the Black extension, configure Black as the default formatter, and add the following line to your `settings.json`:

```json
{
    "editor.formatOnSave": true
}
```

### Collision with Flake8
Black and Flake8 can sometimes have conflicting style rules. To avoid this, we have configured Flake8 to allow double quotes, which is the default style used by Black. This configuration is set in the `ament_flake8.ini` file, which is part of the ROS 2 ament_flake8 package. See the `.devcontainer/Dockerfile` for details on how this is set up in the development environment.