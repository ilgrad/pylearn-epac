=================================
Memory testing
=================================

Workflow
--------

CV / Methods [SVC (kernel="linear") , SVC (kernel="rbf")]

With memory mapping, with 16GB available
----------------------------------------

Command::

```
    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5]
```

```
    n_features | Size X | Size Y |    1 process     |    2 processes   |    3 processes   |
      50000    |  100MB |   2kB  |   621MB |   85s  |  1361MB |   58s  |  1974MB |   55s  |
     100000    |  200MB |   2kB  |  1208MB |  145s  |  2631MB |  105s  |  3830MB |  105s  |
     200000    |  400MB |   2kB  |  2381MB |  286s  |  6536MB |  216s  |  7541MB |  236s  |
     400000    |  800MB |   2kB  |  4728MB |  577s  | 13080MB |  538s  | 15722MB |  672s  |
     800000    | 1600MB |   2kB  |  9424MB | 1219s  |   MEMORY ERROR   |   MEMORY ERROR   |
    1600000    | 3200MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
```

```
    n_features | Size X | Size Y |    4 processes   |    5 processes   |
      50000    |  100MB |   2kB  |  1875MB |   65s  |  2351MB |   70s  |
     100000    |  200MB |   2kB  |  3597MB |  115s  |  4409MB |  131s  |
     200000    |  400MB |   2kB  |  7050MB |  267s  |  8999MB |  282s  |
     400000    |  800MB |   2kB  | 15061MB |  650s  |   MEMORY ERROR   |
     800000    | 1600MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |
    1600000    | 3200MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |
```
Without memory mapping, with 16GB available
-------------------------------------------

Command::

```
    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5, 6]
```

```
    n_features |    1 process     |    2 processes   |    3 processes   |
      50000    |   688MB |   75s  |  2509MB |   85s  |  3483MB |   95s  |
     100000    |  1340MB |  145s  |  3902MB |  155s  |  5596MB |  176s  |
     200000    |  2646MB |  155s  |   MEMORY ERROR   |   MEMORY ERROR   |
     400000    |  5792MB |  597s  | 20037MB |  740s  | 22031MB |  977s  |
     800000    | 10480MB | 1269s  |   MEMORY ERROR   |   MEMORY ERROR   |
    1600000    |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
```

```
    n_features |    4 processes   |    5 processes   |    6 processes   |
      50000    |  3082MB |  105s  |  3827MB |   70s  |  3528MB |  136s  |
     100000    |  6019MB |  206s  |  7480MB |  131s  |  7597MB |  277s  |
     200000    |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
     400000    | 21546MB | 1026s  |   MEMORY ERROR   |   MEMORY ERROR   |
     800000    |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
    1600000    |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
```

With memory mapping, on the cluster
-----------------------------------

Command::

```
    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5, 6, 7, 8]
```

```
    n_features | Size X | Size Y |  1 process   |  2 processes |  3 processes |  4 processes |
      50000    |  100MB |   2kB  |     234s     |     180s     |     148s     |     143s     |
     100000    |  200MB |   2kB  |     416s     |     318s     |     248s     |     226s     |
     200000    |  400MB |   2kB  |     820s     |     631s     |     542s     |     429s     |
     400000    |  800MB |   2kB  |    1665s     |    1353s     |    1212s     |     803s     |
     800000    | 1600MB |   2kB  |    4530s     |    3235s     |    2466s     |    1913s     |
    1600000    | 3200MB |   2kB  | MEMORY ERROR |    6485s     |    5328s     |    7945s     |
    3200000    | 6400MB |   2kB  | MEMORY ERROR |   7647s ?    |   7643s ?    |    9538s     |
```

```
    n_features | Size X | Size Y |  5 processes |  6 processes |  7 processes |  8 processes |
      50000    |  100MB |   2kB  |     146s     |       0s     |       0s     |       0s     |
     100000    |  200MB |   2kB  |     237s     |       0s     |       0s     |       0s     |
     200000    |  400MB |   2kB  |     465s     |       0s     |       0s     |       0s     |
     400000    |  800MB |   2kB  |    1099s     |       0s     |       0s     |       0s     |
     800000    | 1600MB |   2kB  |    2072s     | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR |
    1600000    | 3200MB |   2kB  |    5188s     | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR |
    1600000    | 3200MB |   2kB  |       0s     | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR |
```

Without memory mapping, on the cluster
--------------------------------------

Command::

```
    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5, 6, 7, 8]
```

```
    n_features |  1 process   |  2 processes |  3 processes |  4 processes |
      50000    |     219s     |       0s     |       0s     |       0s     |
     100000    |     424s     |     314s     |       0s     |       0s     |
     200000    |     731s     |     570s     |       0s     |       0s     |
     400000    |    1452s     |    1117s     |       0s     |       0s     |
     800000    |    3558s     |    3148s     | MEMORY ERROR | MEMORY ERROR |
    1600000    | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR |
```

```
    n_features |  5 processes |  6 processes |  7 processes |  8 processes |
      50000    |       0s     |       0s     |       0s     |       0s     |
     100000    |       0s     |       0s     |       0s     |       0s     |
     200000    |       0s     |       0s     |       0s     |       0s     |
     400000    |       0s     |       0s     |       0s     |       0s     |
     800000    |       0s     | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR |
    1600000    | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR | MEMORY ERROR |
```
