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
#    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5]
```

```
    n_features | Size X | Size Y |    1 process     |    2 processes   |    3 processes
      50000    |  100MB |   2kB  |   621MB |   70s  |  1361MB |   50s  |  1974MB |   50s
     100000    |  200MB |   2kB  |  1208MB |  140s  |  2631MB |  100s  |  3829MB |  100s
     200000    |  400MB |   2kB  |  2381MB |  280s  |  6077MB |  211s  |  7540MB |  231s
     400000    |  800MB |   2kB  |  4728MB |  572s  | 13080MB |  533s  | 15830MB |  657s
     800000    | 1600MB |   2kB  |  9424MB | 1214s  |   MEMORY ERROR   |   MEMORY ERROR
    1600000    | 3200MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR
```

```
    n_features | Size X | Size Y |    4 processes   |    5 processes
      50000    |  100MB |   2kB  |  1871MB |   50s  |  2351MB |   70s
     100000    |  200MB |   2kB  |  3599MB |  100s  |  4409MB |  131s
     200000    |  400MB |   2kB  |  7044MB |  231s  |  8999MB |  282s
     400000    |  800MB |   2kB  | 15534MB |  657s  |   MEMORY ERROR   
     800000    | 1600MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   
    1600000    | 3200MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   
```
With memory mapping, with 16GB available
----------------------------------------

Command::

```
#    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5]
```

```
    n_features | Size X | Size Y |    1 process     |    2 processes   |    3 processes
      50000    |  100MB |   2kB  |   688MB |   75s  |  2509MB |   85s  |  3483MB |   95s
     100000    |  200MB |   2kB  |  1340MB |  145s  |  3902MB |  155s  |  5596MB |  176s
     200000    |  400MB |   2kB  |  2646MB |  155s  |   MEMORY ERROR   |   MEMORY ERROR
     400000    |  800MB |   2kB  |  5792MB |  597s  | 20037MB |  740s  | 22031MB |  977s
     800000    | 1600MB |   2kB  | 10480MB | 1269s  |   MEMORY ERROR   |   MEMORY ERROR
    1600000    | 3200MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR
```

```
    n_features | Size X | Size Y |    4 processes   |    5 processes   |    6 processes
      50000    |  100MB |   2kB  |  3082MB |  105s  |  3827MB |   70s  |  3528MB |  136s
     100000    |  200MB |   2kB  |  6019MB |  206s  |  7480MB |  131s  |  7597MB |  277s
     200000    |  400MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR
     400000    |  800MB |   2kB  | 21546MB | 1026s  |   MEMORY ERROR   |   MEMORY ERROR
     800000    | 1600MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR
    1600000    | 3200MB |   2kB  |   MEMORY ERROR   |   MEMORY ERROR   |   MEMORY ERROR
...







