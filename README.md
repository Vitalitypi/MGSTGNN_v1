# train steps

### 1、generate data

`

```bash
cd generate
python generate.py
```

`

### 2、training

```bash
python -u main.py --gpu_id=0 2>&1 | tee exps/PEMS04.log
```

