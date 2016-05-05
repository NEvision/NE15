NE15-MNIST Database
===================
NE15-MNIST contains four sub datasets:    
**Poissonian:**   
The code and example for generating Poissonian spikes from MNIST is located in the folder *Poissonian*.   
**Focol Rank Code Order:**   
The code and example for generating spikes from MNIST using Focol is located in the folder *Focol*.  
**DVS recorded flashing MNIST digits:**   
download from: https://goo.gl/ru0fXP   
**DVS recorded moving MNIST digits:**     
download from: http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html  

You are welcome to cite the paper if you use the database.    
"Benchmarking Spike-Based Visual Recognition: a Dataset and Evaluation",    
Qian Liu, Garibaldi Pineda Garca, Evangelos Stromatias,Teresa Gotarredona, and Steve Furber   


Benchmark 1: Supervised Online STDP
===================================
**Input:** Poisson spikes: NE15-Poissonian  
**Network:** Simple one layer network, fully connected decision layer(50x10), current based LIF-exp neuron  
**Training:** K-means clusters as preprocessing, Supervised STDP, 18,000 biological second  
**Testing:** 5k Hz input rate, 1s per test and 200ms interval silence (in biological time)  
**Performance:** 92.99% accuracy, 13.82ms latency, 4.17M Sopbs  

| Hardware Platform                                                                                          | Accuracy (%) | Sim Time (s) | Enery (KJ)  | Ref                                                                         |
|:----------------------------------------------------------------------------------------------------------:| ------------:| ------------:| -----------:|  ---------------------------------------------------------------------------:|
| [SpiNNaker](http://ieeexplore.ieee.org/xpl/abstractAuthors.jsp?arnumber=6750072) | 92.99 | 12,000 | 4.92 | |
