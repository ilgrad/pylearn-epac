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
    n_features | Size X | Size Y |    1 process    |    2 processes  |    3 processes  |    4 processes  |
      50000    |  100MB |   2kB  |   621MB |   70s |  1361MB |   50s |  2263MB |   55s |  1872MB |   60s |
     100000    |  200MB |   2kB  |  1208MB |  140s |  2877MB |  100s |  4526MB |  106s |  3595MB |  116s |
     200000    |  400MB |   2kB  |  2381MB |  286s |  5885MB |  211s |  7941MB |  243s |  7049MB |  269s |
     400000    |  800MB |   2kB  |  4729MB |  598s | 16245MB |  566s | 15663MB |  683s | 15094MB |  660s |
     800000    | 1600MB |   2kB  | 10928MB | 1248s |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
    1600000    | 3200MB |   2kB  |   MEMORY ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
```

```
    n_features | Size X | Size Y |    5 processes  |    6 processes  |    7 processes  |    8 processes  |
      50000    |  100MB |   2kB  |  2291MB |   65s |  2534MB |   71s |  2447MB |   71s |  2546MB |   71s |
     100000    |  200MB |   2kB  |  4511MB |  142s |  5367MB |  147s |  5528MB |  132s |  4693MB |  147s |
     200000    |  400MB |   2kB  |  9663MB |  325s | 10157MB |  352s |  9299MB |  352s | 10237MB |  311s |
     400000    |  800MB |   2kB  |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |
     800000    | 1600MB |   2kB  |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
    1600000    | 3200MB |   2kB  |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
```
Without memory mapping, with 16GB available
-------------------------------------------

Command::

```
    . epac/tests/launch_test_memm.sh
    n_samples = 500
    n_features = [50000, 100000, 200000, 400000, 800000, 1600000]
    num_processes  = [1, 2, 3, 4, 5, 6, 7, 8]
```

```
    n_features |    1 process    |    2 processes  |    3 processes  |    4 processes  |
      50000    |   687MB |   80s |  1982MB |   75s |  3228MB |   90s |  3240MB |  106s |
     100000    |  1340MB |  145s |  4445MB |  151s |  5967MB |  174s |  6019MB |  207s |
     200000    |  2646MB |  296s |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
     400000    |  5258MB |  608s | 18823MB |  761s | 23507MB |  971s | 23101MB | 1051s |
     800000    | 10483MB | 1303s |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
    1600000    |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |
```

```
    n_features |    5 processes  |    6 processes  |    7 processes  |    8 processes  |
      50000    |  3827MB |  126s |  4355MB |  137s |  4191MB |  136s |  4311MB |  136s |
     100000    |  8830MB |  253s |  8167MB |  279s |  8270MB |  279s |  8300MB |  279s |
     200000    |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
     400000    |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |
     800000    |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |   SYSTEM ERROR  |
    1600000    |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |   MEMORY ERROR  |
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
    n_features | Size X | Size Y |    1 process    |  2 processes              |          3 processes                |
               |                 | 1st job | Time  | 1st job | 2nd job | Time  | 1st job | 2nd job | 3rd job | Time  |
      50000    |  100MB |   2kB  |   639MB |  223s |   638MB |   638MB |  153s |   572MB |   505MB |   506MB |  126s |
     100000    |  200MB |   2kB  |  1240MB |  411s |  1239MB |  1239MB |  282s |   973MB |   971MB |   971MB |  236s |
     200000    |  400MB |   2kB  |  2442MB |  888s |  2440MB |  2440MB |  765s |  2437MB |  2437MB |  2173MB |  477s |
     400000    |  800MB |   2kB  |  4845MB | 1599s |  4842MB |  4842MB | 1316s |  4845MB |  4835MB |  4835MB | 1042s |
     800000    | 1600MB |   2kB  |  9651MB | 3292s | 11451MB |  9645MB | 2914s |  9632MB |  9632MB |  9651MB | 2339s |
    1600000    | 3200MB |   2kB  | 19264MB | 7812s | 19252MB | 19252MB | 5890s | 19226MB | 19226MB | 19264MB | 4104s |
    3200000    | 3200MB |   2kB  |   MEMORY ERROR  |        MEMORY ERROR       |             MEMORY ERROR            |
```
```
    n_features | Size X | Size Y |                    4 processes                |
               |                 | 1st job | 2nd job | 3rd job | 4th job | Time  |
      50000    |  100MB |   2kB  |   506MB |   505MB |   372MB |     0MB |  101s |
     100000    |  200MB |   2kB  |   838MB |   838MB |   971MB |   838MB |  193s |
     200000    |  400MB |   2kB  |  1907MB |  1907MB |  1904MB |  1904MB |  342s |
     400000    |  800MB |   2kB  |  3776MB |  3770MB |  3776MB |  3770MB |  891s |
     800000    | 1600MB |   2kB  |  7501MB |  7501MB |  7514MB |  7514MB | 1712s |
    1600000    | 3200MB |   2kB  | 18114MB | 14963MB | 14989MB | 14963MB | 3601s |
    3200000    | 3200MB |   2kB  |                  MEMORY ERROR                 |
```
```
    n_features | Size X | Size Y |                          5 processes                    |
               |                 | 1st job | 2nd job | 3rd job | 4th job | 5th job | Time  |
      50000    |  100MB |   2kB  |   438MB |   438MB |   505MB |   505MB |   506MB |  127s |
     100000    |  200MB |   2kB  |  1240MB |   971MB |   971MB |   971MB |   973MB |  234s |
     200000    |  400MB |   2kB  |  1904MB |  1904MB |  2440MB |  1904MB |  1904MB |  434s |
     400000    |  800MB |   2kB  |  3776MB |  3770MB |  3770MB |  4842MB |  3770MB |  942s |
     800000    | 1600MB |   2kB  |  7501MB |  7514MB |  7501MB |  9645MB |  7501MB | 1900s |
    1600000    | 3200MB |   2kB  | 19251MB |  9587MB | 12062MB | 10329MB | 12381MB | 8350s |
    3200000    | 3200MB |   2kB  |                       MEMORY ERROR                      |
```
```
    n_features | Size X | Size Y |                            6 processes                            |
               |                 | 1st job | 2nd job | 3rd job | 4th job | 5th job | 6th job | Time  |
      50000    |  100MB |   2kB  |   372MB |   438MB |   438MB |   438MB |   438MB |     0MB |   90s |
     100000    |  200MB |   2kB  |   971MB |   706MB |   706MB |   705MB |   838MB |   838MB |  150s |
     200000    |  400MB |   2kB  |  1904MB |  1904MB |  1904MB |  1907MB |  1907MB |  1904MB |  352s |
     400000    |  800MB |   2kB  |  3776MB |  3776MB |  3770MB |  3770MB |  3770MB |  3770MB |  644s |
     800000    | 1600MB |   2kB  |  7501MB |  7514MB |  7501MB |  7501MB |  7514MB |  7501MB | 1675s |
    1600000    | 3200MB |   2kB  | 13254MB | 13304MB | 13475MB | 11402MB | 14963MB | 14963MB |11756s |
    3200000    | 3200MB |   2kB  |                            MEMORY ERROR                           |
```
```
    n_features | Size X | Size Y |                     7 processes (only 6 jobs)                     |
               |                 | 1st job | 2nd job | 3rd job | 4th job | 5th job | 6th job | Time  |
      50000    |  100MB |   2kB  |   438MB |   438MB |   438MB |   506MB |   506MB |   505MB |   98s |
     100000    |  200MB |   2kB  |   705MB |   705MB |   838MB |   838MB |   706MB |   838MB |  154s |
     200000    |  400MB |   2kB  |  1904MB |  1904MB |  1907MB |  1904MB |  1904MB |  1907MB |  355s |
     400000    |  800MB |   2kB  |  3770MB |  3776MB |  3776MB |  3770MB |  3770MB |  3770MB |  631s |
     800000    | 1600MB |   2kB  |  7514MB |  7501MB |  7514MB |  7501MB |  7501MB |  7501MB | 1443s |
    1600000    | 3200MB |   2kB  |  8864MB |  8175MB |  9110MB |  8527MB |  9906MB | 14989MB |20223s |
    3200000    | 3200MB |   2kB  |                            MEMORY ERROR                           |
```
```
    n_features | Size X | Size Y |                     8 processes (only 6 jobs)                     |
               |                 | 1st job | 2nd job | 3rd job | 4th job | 5th job | 6th job | Time  |
      50000    |  100MB |   2kB  |   438MB |   505MB |   505MB |   438MB |   505MB |   505MB |   94s |
     100000    |  200MB |   2kB  |   973MB |   971MB |   971MB |   971MB |   971MB |   973MB |  150s |
     200000    |  400MB |   2kB  |  1904MB |  1904MB |  1904MB |  1907MB |  1907MB |  1904MB |  350s |
     400000    |  800MB |   2kB  |  3770MB |  3770MB |  3770MB |  3776MB |  3770MB |  3776MB |  643s |
     800000    | 1600MB |   2kB  |  7514MB |  7501MB |  7514MB |  7501MB |  7501MB |  7501MB | 1404s |
    1600000    | 3200MB |   2kB  | 12262MB | 10702MB | 12572MB | 12259MB | 12080MB | 12262MB | 4641s |
    3200000    | 3200MB |   2kB  |                            MEMORY ERROR                           |
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
    n_features |    1 process    |  2 processes              |          3 processes                |
               | 1st job | Time  | 1st job | 2nd job | Time  | 1st job | 2nd job | 3rd job | Time  |
      50000    |   705MB |  216s |   704MB |   706MB |  145s |   572MB |   571MB |   572MB |  137s |
     100000    |  1373MB |  319s |  1371MB |  1373MB |  299s |  1373MB |  1370MB |  1373MB |  282s |
     200000    |  2707MB |  866s |  2706MB |  2706MB |  561s |  2704MB |  2704MB |  2707MB |  455s |
     400000    |  5376MB | 1660s |  5373MB |  5373MB | 1140s |  5370MB |  5376MB |  5370MB | 1001s |
     800000    | 10714MB | 3457s | 10708MB | 10708MB | 2310s | 10701MB | 10701MB | 10714MB | 2479s |
    1600000    | 21364MB | 7278s | 16980MB | 13586MB | 8641s | 16606MB | 17525MB | 18908MB | 5902s |
    3200000    |   MEMORY ERROR  |        MEMORY ERROR       |             MEMORY ERROR            |
```
```
    n_features |                    4 processes                |
               | 1st job | 2nd job | 3rd job | 4th job | Time  |
      50000    |   572MB |   572MB |   572MB |     0MB |  104s |
     100000    |  1106MB |  1106MB |  1106MB |  1106MB |  217s |
     200000    |  2171MB |  2173MB |  2173MB |  2171MB |  413s |
     400000    |  4304MB |  4308MB |  4304MB |  4307MB |  862s |
     800000    |  8576MB |  8570MB |  8576MB |  8570MB | 1976s |
    1600000    | 11628MB | 17086MB | 10058MB | 17114MB | 5357s |
    3200000    |                  MEMORY ERROR                 |
```
```
    n_features |                          5 processes                    |
               | 1st job | 2nd job | 3rd job | 4th job | 5th job | Time  |
      50000    |   572MB |   705MB |   571MB |   572MB |   572MB |  121s |
     100000    |  1371MB |  1106MB |  1106MB |  1104MB |  1106MB |  242s |
     200000    |  2171MB |  2706MB |  2171MB |  2171MB |  2173MB |  468s |
     400000    |  5373MB |  4303MB |  4304MB |  4307MB |  4304MB | 1010s |
     800000    |  8570MB |  8576MB | 10708MB |  8570MB |  8570MB | 2170s |
    1600000    | 13068MB | 21349MB | 15769MB | 12900MB | 17114MB | 5133s |
    3200000    |                       MEMORY ERROR                      |
```
```
    n_features |                                  6 processes                      |
               | 1st job | 2nd job | 3rd job | 4th job | 5th job | 6th job | Time  |
      50000    |   572MB |   571MB |   572MB |   572MB |   572MB |     0MB |   97s |
     100000    |  1104MB |  1106MB |  1106MB |  1104MB |  1106MB |  1106MB |  195s |
     200000    |  2171MB |  2171MB |  2173MB |  2171MB |  2173MB |  2171MB |  348s |
     400000    |  4304MB |  4304MB |  4307MB |  4304MB |  4307MB |  4304MB |  908s |
     800000    |  8576MB |  8570MB |  8570MB |  8570MB |  8570MB |  8576MB | 1607s |
    1600000    | 17114MB | 17101MB | 14096MB | 15647MB | 12592MB | 13655MB | 3408s |
    3200000    |                            MEMORY ERROR                           |
```
```
    n_features |                     7 processes (only 6 jobs)                     |
               | 1st job | 2nd job | 3rd job | 4th job | 5th job | 6th job | Time  |
      50000    |   572MB |   571MB |   572MB |   572MB |   572MB |     0MB |   97s |
     100000    |  1104MB |  1106MB |  1106MB |  1106MB |  1106MB |  1104MB |  196s |
     200000    |  2173MB |  2171MB |  2171MB |  2171MB |  2171MB |  2173MB |  339s |
     400000    |  4307MB |  4304MB |  4307MB |  4304MB |  4304MB |  4304MB |  817s |
     800000    |  8576MB |  8570MB |  8576MB |  8570MB |  8570MB |  8570MB | 1640s |
    1600000    | 13768MB | 17101MB | 14841MB | 17114MB | 12559MB | 14369MB | 3748s |
    3200000    |                            MEMORY ERROR                           |
```
```
    n_features |                     8 processes (only 6 jobs)                     |
               | 1st job | 2nd job | 3rd job | 4th job | 5th job | 6th job | Time  |
      50000    |   572MB |   571MB |   572MB |   572MB |   572MB |   571MB |   98s |
     100000    |  1106MB |  1106MB |  1104MB |  1106MB |  1104MB |  1106MB |  202s |
     200000    |  2173MB |  2173MB |  2171MB |  2171MB |  2171MB |  2171MB |  339s |
     400000    |  4304MB |  4304MB |  4304MB |  4304MB |  4307MB |  4307MB |  780s |
     800000    |  8576MB |  8570MB |  8570MB |  8570MB |  8570MB |  8576MB | 1538s |
    1600000    | 10375MB | 13456MB | 13368MB | 12271MB | 12881MB |  8415MB | 7246s |
    3200000    |                            MEMORY ERROR                           |
```
