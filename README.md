## wheeled_bipedal_gym

> Mainly based on https://github.com/clearlab-sustech/Wheel-Legged-Gym, some reward functions were modified and models were replaced, and the `DirectDriveTech Diablo Robot` model was used for training.

### Installation

1. Create a new python virtual env with python 3.8 
2. Install pytorch with cuda from https://pytorch.org/get-started/
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install wheel_legged_gym
   - Clone this repository
   - `cd wheeled-bipedal-gym && pip install -e .`

### Usage

#### train

```
python wheeled_bipedal_gym/scripts/train.py --task=diablo
```

It takes about three thousand iterations to reach convergence.

#### play

```
python wheeled_bipedal_gym/scripts/play.py --task=diablo
```

