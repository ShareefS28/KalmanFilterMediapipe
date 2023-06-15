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