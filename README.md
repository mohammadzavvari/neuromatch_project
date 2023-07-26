# neuromatch_project

## Maddpg-mpe-pytorch:

- If you want to run maddpg-mpe-pytorch in the simple-tag mode without using command-line, just change  the main.py, parser.add_argument() to the lines  below:(Add "--" before "env", add "default='simple_tag'")
  ```
  parser.add_argument(
      "--env",
      default="simple_tag",
      type=str,
      help="name of the environment",
      choices=[
          "simple_adversary",
          "simple_crypto",
          "simple_push",
          "simple_reference",
          "simple_speaker_listener",
          "simple_spread",
          "simple_tag",
          "simple_world_comm",
      ],
  )

  ```

* Also to run  the  code without dependency issues, run:
  ```
  pip uninstall gym
  pip install gym==0.10.5  
  ```
