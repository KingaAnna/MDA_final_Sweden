#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from py2neo import Graph
from py2neo import Node
from py2neo import Relationship
import pandas as pd
import numpy as np


# In[ ]:


class my_graph:
    def __init__(self,data,period,my_list):
        self.data=data
        self.place=data['state']
        self.date=data['date']
        self.period=period
        self.my_list=my_list
    def __repr__(self):
        return "Makes nodes and edges based on the correlations between timeseries for each state"
    def fit_transform(self):
        data_grouped=self.data.groupby(["state", "date"],as_index=False).agg({self.my_list[1]:'sum'})
        #If a date is not present in a certain county, we can assume that no cases where reported yet during this period
        #(the NA values occur for early dates, when no Covid case was reported yet in that county).
        rolling_pivot=pd.pivot_table(data=data_grouped,index='date', columns=['state'], values=self.my_list[1])
        rolling_pivot.fillna(0,inplace=True)
        correlation=rolling_pivot.corr()
        correlation.index.names = ['source_place']
        np.fill_diagonal(correlation.values, 0)
        #We can construct a graph in Neo4j using the correlation matrix column names as nodes. 
        #Two nodes are connected if their entry in the correlation matrix is larger than 0.7 (weak correlations are ignored).
        stacked=correlation[correlation>0.7].stack().reset_index().rename(columns={'state':'target_place',0:'correlation'})
        stacked=stacked.rename(columns={'source_place':'source','target_place':'target'})
        #The file with relationships is named correlation_states_period.csv
        stacked.to_csv('Data_input_neo4j/correlation_states_'+self.my_list[0]+'_'+self.period+'.csv',index=False)
        #The file with nodes is named nodes_states_period.csv
        correlation.index.to_frame().to_csv('Data_input_neo4j/nodes_states_'+self.my_list[0]+'.csv',index=False)


# In[ ]:


class my_algorithms:
    def __init__(self,graph):
        self.graph=graph
    def fit_community(self,nodes,relationships):
        self.graph.run("MATCH (n) DETACH DELETE n")
        
        # make nodes
        query="""LOAD CSV WITH HEADERS FROM $file as row
        with row
        CALL apoc.create.node(['State','Place'],{name:row.source_place}) YIELD node
        RETURN distinct true"""
        self.graph.run(query,file=nodes)
        
 
        # make relationships
        query="""LOAD CSV WITH HEADERS FROM $file as row
        with row
        MATCH (source:State{name:row.source})
        MATCH (target:State{name:row.target})
        CALL apoc.create.relationship(source,"CORRELATED",{correlation:toFloat(row.correlation)},target) YIELD rel
        RETURN distinct true"""
        self.graph.run(query,file=relationships)

        # Create an in-memory graph
        query="""CALL gds.graph.create('Covid','State',{CORRELATED:{properties:'correlation', orientation:'UNDIRECTED'}})"""
        self.graph.run(query)

        # Perform community detection
        query="""CALL gds.labelPropagation.stream('Covid',{maxIterations: 10,relationshipWeightProperty:'correlation'})
        YIELD nodeId, communityId
        RETURN communityId AS Label,
        collect(gds.util.asNode(nodeId).name) AS Places
        ORDER BY size(Places) DESC"""
        result=self.graph.run(query).data()
        df=pd.DataFrame(result)
        df['cluster']=df.index
        df.drop(columns=['Label'],inplace=True)

        # Delete the in-memory graph
        self.graph.run("CALL gds.graph.drop('Covid')")

        return df
    

