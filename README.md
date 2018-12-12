## algosel-rl

Source code for my Master's dissertation entitled _Algorithm Selection for Subgraph Isomorphism Problems: A Reinforcement Learning Approach_.

### Running the scripts

To start, download and install R (version 3.4.4+) from [CRAN](https://cloud.r-project.org). This installation contains the R interpreter and a simple GUI app for creating R scripts. This is sufficient to run the scripts in this repo; however, if you are planning to debug or modify the scripts, I highly suggest to use a full-featured IDE like [RStudio](https://www.rstudio.com/products/rstudio/download/). 

#### Installing prerequisite packages

Run ```source('install_packages.R')``` on R command line to install all the necessary packages.

#### Rendering R Markdown (.Rmd) files

The rendered contents of the .Rmd files can be readily viewed at [RPubs](https://rpubs.com). Check out the following links:

* [ntbk_eda_graphs2015.Rmd](http://rpubs.com/kvrigor/eda_graphs)
* [ntbk_eda_hard.Rmd](http://rpubs.com/kvrigor/eda_graphs_hard)
* [ntbk_asresults_graphs2015.Rmd](http://rpubs.com/kvrigor/asresults_graphs)
* [ntbk_asresults_reinforce.Rmd](http://rpubs.com/kvrigor/asresults_reinforce)

These files can be rendered locally, and the easiest way to do this is through RStudio. Check out this [guide](https://rmarkdown.rstudio.com/articles_intro.html).

### Dissertation paper

The paper was written in [LaTeX](https://en.wikibooks.org/wiki/LaTeX) using [TeXStudio](https://www.texstudio.org) software on Windows. Typesetting files are taken from [utmthesis](https://github.com/utmthesis/utmthesis/releases/tag/v5.1) (v5.1) GitHub repository.

### Useful Links

**R Packages**
* Algorithm Selection Library (aslib).  [RDoc](https://www.rdocumentation.org/packages/aslib/versions/0.1) | [GitHub](https://www.rdocumentation.org/packages/aslib/versions/0.1) 
* Leveraging Learning to Automatically Manage Algorithms (llama).  [RDoc](https://www.rdocumentation.org/packages/llama/versions/0.9.2) | [BitBucket](https://bitbucket.org/lkotthoff/llama) 
* R interface to TensorFlow.  [link](https://tensorflow.rstudio.com/tensorflow/) 

**Recommended Reads**
* Kotthoff, L. (2016). **Algorithm selection for combinatorial search problems: A survey**. In Data Mining and Constraint Programming (pp. 149-190). Springer, Cham. [paper](http://www.aaai.org/ojs/index.php/aimagazine/article/download/2460/2438)
* Kotthoff, L., McCreesh, C., & Solnon, C. (2016, May). **Portfolios of subgraph isomorphism algorithms**. In International Conference on Learning and Intelligent Optimization (pp. 107-122). Springer, Cham. [paper](https://hal.archives-ouvertes.fr/hal-01301829/document)
* Bischl, B., Kerschke, P., Kotthoff, L., Lindauer, M., Malitsky, Y., Fréchette, A., ... & Vanschoren, J. (2016). **Aslib: A benchmark library for algorithm selection**. Artificial Intelligence, 237, 41-58. [paper](https://arxiv.org/pdf/1506.02465)
* Kotthoff, L. (2013). **LLAMA: leveraging learning to automatically manage algorithms**. arXiv preprint arXiv:1306.1031. [paper](https://arxiv.org/pdf/1306.1031)
* Lindauer, M., van Rijn, J. N., & Kotthoff, L. (2017, December). **Open Algorithm Selection Challenge 2017: Setup and Scenarios**. In Open Algorithm Selection Challenge 2017 (pp. 1-7). [paper](http://proceedings.mlr.press/v79/lindauer17a/lindauer17a.pdf)
* Smith-Miles, K. A. (2009). **Cross-disciplinary perspectives on meta-learning for algorithm selection**. ACM Computing Surveys (CSUR), 41(1), 6. [paper](https://www.researchgate.net/profile/Kate_Smith-Miles/publication/220565856_Cross-Disciplinary_Perspectives_on_Meta-Learning_for_Algorithm_Selection/links/57e1f8d208ae1f0b4d93fa7d/Cross-Disciplinary-Perspectives-on-Meta-Learning-for-Algorithm-Selection.pdf)
* Sutton, R. S., & Barto, A. G. (1998). **Introduction to reinforcement learning** (Vol. 135). Cambridge: MIT press. [book](http://incompleteideas.net/book/bookdraft2017nov5.pdf)
* Policy Gradients
    * I still find Sutton & Barto's **Introduction to Reinforcement Learning** the easiest to understand regarding this topic. It might be understood better if complemented with readings from other sources. Check these slides:
    [1](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf)
    [2](https://www.ias.informatik.tu-darmstadt.de/uploads/Research/MPI2007/MPI2007peters.pdf)
    [3](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf)

**Others**
* ASlib website.  [link](http://www.coseal.net/aslib/) 
* GRAPHS-2015 dataset source.  [GitHub](https://github.com/coseal/aslib_data/tree/master/GRAPHS-2015) 
* Reinforcement Learning study plan. [introductory](https://github.com/dennybritz/reinforcement-learning) | [deep RL](https://www.reddit.com/r/reinforcementlearning/comments/6w8kz1/d_study_group_for_deep_rl_policy_gradient_methods/)
