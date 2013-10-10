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
    n_features | Size X | Size Y |    4 processes   |    5 processes   |    6 processes   |
      50000    |  100MB |   2kB  |  1875MB |   65s  |  2351MB |   70s  |  2451MB |   75s  |
     100000    |  200MB |   2kB  |  3597MB |  115s  |  4409MB |  131s  |  4695MB |  146s  |
     200000    |  400MB |   2kB  |  7050MB |  267s  |  8999MB |  282s  |  9700MB |  358s  |
     400000    |  800MB |   2kB  | 15061MB |  650s  |   MEMORY ERROR   |   MEMORY ERROR   |
     800000    | 1600MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
    1600000    | 3200MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
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
      50000    |   687MB |   75s  |  1983MB |   85s  |  3067MB |   95s  |
     100000    |  1340MB |  145s  |  4740MB |  155s  |  5596MB |  176s  |
     200000    |  2646MB |  296s  |   SYSTEM ERROR   |   SYSTEM ERROR   |
     400000    |  5787MB |  597s  | 18859MB |  740s  | 22031MB |  977s  |
     800000    | 10480MB | 1269s  |   SYSTEM ERROR   |   SYSTEM ERROR   |
    1600000    |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
```

```
    n_features |    4 processes   |    5 processes   |    6 processes   |
      50000    |  3083MB |  105s  |  3827MB |  125s  |  3528MB |  136s  |
     100000    |  6019MB |  206s  |  7527MB |  256s  |  7119MB |  277s  |
     200000    |   SYSTEM ERROR   |   SYSTEM ERROR   |   SYSTEM ERROR   |
     400000    | 21546MB | 1033s  | 23146MB | 1119s  |   MEMORY ERROR   |
     800000    |   SYSTEM ERROR   |   SYSTEM ERROR   |   MEMORY ERROR   |
    1600000    |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR   |
```

SYSTEM ERROR : SystemError: error return without exception set

With memory mapping, on the cluster
-----------------------------------

Command::

```
    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000, 3200000]
    num_processes  = [1, 2, 3, 4, 5, 6]
```

```
    n_features | Size X | Size Y |  1 process   |  2 processes |  3 processes |
      50000    |  100MB |   2kB  |     234s     |     180s     |     148s     |
     100000    |  200MB |   2kB  |     416s     |     318s     |     248s     |
     200000    |  400MB |   2kB  |     820s     |     631s     |     542s     |
     400000    |  800MB |   2kB  |    1665s     |    1353s     |    1212s     |
     800000    | 1600MB |   2kB  |    4530s     |    3235s     |    2466s     |
    1600000    | 3200MB |   2kB  |   7407s ?    |    6485s     |    5328s     |
    3200000    | 6400MB |   2kB  |       0s     |   7647s ?    |   7643s ?    |
```

```
    n_features | Size X | Size Y |  4 processes |  5 processes |  6 processes |
      50000    |  100MB |   2kB  |     143s     |     146s     |     122s     |
     100000    |  200MB |   2kB  |     226s     |     237s     |     202s     |
     200000    |  400MB |   2kB  |     429s     |     465s     |     379s     |
     400000    |  800MB |   2kB  |     803s     |    1099s     |     796s     |
     800000    | 1600MB |   2kB  |    1913s     |    2072s     |    1509s     |
    1600000    | 3200MB |   2kB  |    7945s     |    5188s     |    3096s     |
    3200000    | 6400MB |   2kB  |    9538s     |   7651s ?    |   7615s ?    |
```

Without memory mapping, on the cluster
--------------------------------------

Command::

```
    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5, 6]
```

```
    n_features |  1 process   |  2 processes |  3 processes |
      50000    |     218s     |     189s     |     137s     |
     100000    |     384s     |     322s     |     227s     |
     200000    |     765s     |     554s     |     450s     |
     400000    |    1522s     |    1167s     |    1240s     |
     800000    |    3567s     |    2715s     |    2419s     |
    1600000    |    9120s     |   7697s ?    |    5760s     |
```

```
    n_features |  4 processes |  5 processes |  6 processes |
      50000    |     128s     |     157s     |     129s     |
     100000    |     207s     |     263s     |     175s     
     200000    |     438s     |     448s     |     352s     |
     400000    |     927s     |    1057s     |     806s     |
     800000    |    2027s     |    2114s     |    1751s     |
    1600000    |    6025s     |    5993s     |    8440s     |
```
