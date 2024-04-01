# Solving Max-Min Fair Resource Allocations Quickly on Large Graphs

`Soroush` is a scalable and general max-min fair allocator. It contains a group of approximate and heuristic methods that allow users to control the trade-offs between efficiency, fairness and speed. For more information, see our NSDI24 paper ([Solving Max-Min Fair Resource Allocations Quickly on Large Graphs](https://www.usenix.org/conference/nsdi24/presentation/namyar-solving)).


## Code Structure
```
├── cluster_scheduling      # Scripts and implementation for the CS usercase.
|           |       
|           ├── alg         # implementation of all the allocators in Soroush.
|           |
|           ├── scripts     # code for generating different problem instances and benchmarking different allocators.
|           |
|           └── utilities   # common utility functions for cluster scheduling. 
|
|
└── traffic_engineering     # Scripts and implementations for the TE usecase
            |
            ├── alg         # implementation of all the allocators in Soroush
            |
            ├── benchmarks  #
            |
            ├── scripts     #
            |
            └── utilities   #
```

### Installation

Please refer to the Readme under `cluster_scheduling` and `traffic_engineering` for problem specific guidelines.
## Citation
```bibtex
@inproceedings{soroush,
  author = {Namyar, Pooria and Arzani, Behnaz and Kandula, Srikanth and Segarra, Santiago and Crankshaw, Daniel and Krishnaswamy, Umesh and Govindan, Ramesh and Raj, Himanshu},
  title = {{S}olving {M}ax-{M}in {F}air {R}esource {A}llocations
  		  {Q}uickly on {L}arge {G}raphs},
  booktitle = {21st USENIX Symposium on Networked Systems Design and
  		  Implementation (NSDI 24)},
  year = {2024},
}
```
