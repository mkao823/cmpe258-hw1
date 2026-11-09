metric: `val_bpb` (lower is better)

```
                 n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_gpt_modern_T256  paired_p
name                                                                                                        
gpt_modern_T512        2        2.1029       0.0008    3.1626     3.6364                   -0.0129    0.0611
gpt_modern_T128        2        2.1135       0.0090    3.1626     3.1936                   -0.0023    0.7969
gpt_modern_T256        2        2.1158       0.0010    3.1626     3.3702                    0.0000       NaN
lstm_T512              2        2.1298       0.0065    1.0698     1.8532                    0.0140    0.2301
lstm_T256              2        2.1371       0.0048    1.0698     1.2006                    0.0213    0.1208
lstm_T128              2        2.1486       0.0030    1.0698     0.8753                    0.0328    0.0280
gpt_modern_T64         2        2.1562       0.0006    3.1626     2.9721                    0.0404    0.0176
lstm_T64               2        2.1806       0.0051    1.0698     0.7310                    0.0647    0.0290
```
