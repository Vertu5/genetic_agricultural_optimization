# -*- coding: utf-8 -*-
import numpy as np

def read_ProdCost(filename, Map=None):
    """
    Reads a grid map text file containing numeric values into a numpy array.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    
    matrix = []
    for line in lines:
        line_str = line.strip()
        if line_str:
            row = [float(char) for char in line_str if char.isdigit()]
            matrix.append(row)
            
    matrix = np.array(matrix, dtype=float)
    return matrix

def Decision_map(data):
    """
    Constructs a DecisionMap matrix where:
      0 = Restricted (R)
      1 = Cultivable candidate (C)
      2 = Existing agricultural land (A)
    """
    if isinstance(data, list):
        Map = np.zeros((len(data), len(data[0])))
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] == "R":
                    Map[i][j] = 0
                elif data[i][j] == "C":
                    Map[i][j] = 1
                else:   
                    Map[i][j] = 2
        return Map
    elif isinstance(data, np.ndarray):
        return data.copy()
    return data

def preprocess(DecisionMap, productivity_map, proximity_map):
    """
    Preprocesses the decision map by filtering candidate cells if needed.
    """
    return DecisionMap.copy()
