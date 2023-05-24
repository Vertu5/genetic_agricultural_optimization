# genetic_agricultural_optimization
Optimize agricultural site selection using genetic algorithms. Maximize productivity, proximity to populated areas, and compactness while respecting budget and land use constraints. Analyze land maps based on cost and production indices to find the ideal location for agricultural expansion. And using Promethee to order the results. 



                                   PROJECT ROOT

                    |-- Data/                             # Contains all datas 
                    |    |-- settings.py                  # Defines Global Settings
                    |    |-- CDB001.py                    # An example of file
                    |                      

                    |-- Saved_parameters/                 # Contains weights and biases obtains after training
                    |    |
                    |    
                    |-- long_term_prediction/             # LSTM for future prediction
                    |         |      
                    |         |-- Saved_parameters/       # Contains weights and biases obtains after training

                    |-- Keras_MODEL.ipynb                 # Keras based LSTM 

                    |-- LSTM_Class.py                     # Declaration of LSTM class

                    |-- requirements.txt                  # Packages

                    |-- short_term_predict.py             # LSTM to predict a short sequence
                    |-- short_term_train.py               # LSTM to train a short sequence

                    |-- ************************************************************************
