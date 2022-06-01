# MDA_final
In our project, we studied how Covid-19 evolved in the USA up to 2022-04-11. The raw datasets that were used can be found in the folder _RawData_, the preprocessed data sets can be found in the folder _PreprocessedData_ and the data that need to be in the input folder of Neo4j can be found in the folder *Data_input_neo4j*. The figures that were obtained during the analysis are in the folder _Figures_. At last, the python files that need to be imported in the notebooks can be found in the folder _Python_files_. The code itself is divided into different notebooks, to keep the code structured. Some notebooks handle the preprocessing of the data, while other notebooks handle to data analysis and visualization. However, they can be run in a random order, since all necessary preprocessed data is already in the folder _PreprocessedData_.

The notebooks are:
  - **CovidMeasuresPreprocessing.ipynb**: Preprocess the data containing Covid measures.
  - **CommutingFlowStatesPreprocessing.ipynb**: Preprocess the commuting flow data set and calculate the degree centrality of a graph, constructed in Neo4j.
  - **Visualization.ipynb**: Visualize the weekly cases in the USA.
  - **infectionRates.ipynb**: Calculate and visualize the infection rates, together with the preprocessed Covid measures data. Also, a SVM model is constructed to predict the infection rates based on training data. 
  - **Clustering_Communities.ipynb**: Construct a graph in Neo4j based on the correlations between cases and deaths and use Label propagation. Build a pipeline including K-means and spectral clustering using the weekly cases.
  - **commuteGraph.ipynb**: Uses workplaceFlows commuting data on commutes from one county to another to calculate centrality based on commuting statistics
  - **Biobot Waste Water.ipynb**: Uses a LM with the virus concentration as the only variable to predict daily increased cases

To get more insight in our analysis and results, we refer to our app on the link https://app-modern-data-analytics.herokuapp.com/.
