metric: `val_bpb` (lower is better)

```
              n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_lstm  paired_p
name                                                                                          
gpt_modern          2        2.1159       0.0010    3.1626     3.8542        -0.0213    0.1222
lstm                2        2.1371       0.0049    1.0698     1.2272         0.0000       NaN
gru                 2        2.1597       0.0017    0.8067     3.6685         0.0226    0.1292
lstm_512            2        2.1692       0.0121    4.2368     2.8697         0.0321    0.2277
gpt_textbook        2        2.1852       0.0016    3.4383     3.5939         0.0481    0.0600
rnn                 2        2.4384       0.0086    0.2803     2.4045         0.3013    0.0201
```
