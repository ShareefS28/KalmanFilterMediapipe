# KalmanFilterMediapipe

## Python
**OpenCV kalmanfilter was a standard kalman (Linear Kalman)**
```bash
python Python/cv_mpHolistic.py
```

#### conda export 
```bash
conda env export --name <env_name> > <file_name>.yml 
```

```bash
conda env export --no-builds --name <env_name> | findstr /V "^prefix: " > <file_name>.yml
```

#### conda import
```bash
conda env create --file <conda_env_file>.yml
```

```bash
conda env create --file environment.yml --name <env_name>
```

<!-- ## C++ -->

<!-- Step | What is happening? <br>
1 | Use f(x) to predict where you go next. <br>
2 | Use jacobian_f() to linearize motion. <br>
3 | Update covariance with prediction uncertainty. <br>
4 | Wait for new measurement z. <br>
5 | Use h(x) to predict how the world looks. <br>
6 | Use jacobian_h() to linearize the camera view. <br>
7 | Calculate the innovation y = z - h(x). <br>
8 | Calculate Kalman Gain to balance trust between prediction and measurement. <br>
9 | Update state using measurement. <br>
10 | Update covariance to reflect new confidence. <br>
11 | Repeat for next frame <br> -->
