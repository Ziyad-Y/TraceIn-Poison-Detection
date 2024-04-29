**Using TraceInCP to Detect Poisoned Data**
![TraceInCP](https://github.com/Ziyad-Y/TraceIn-Posion-Detection/blob/main/tracincp.png)

## Poison ##  
select the poison in main.py

```python
  r.random_label_poison(train_points,0.20) #or 
  r.target_label_poison(train_points,3,9) #or
  r.clean_label_poison(train_points,9,3)
```
