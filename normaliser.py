import json
import pandas as pd
import numpy as np

import textdistance as dst


class AttributeNormaliser():

    def __init__(self, df: pd.DataFrame, data: list) -> None:
        self.values = list(set(df['Attribute']))
        self.df = df
        self.data = data
        self.similarities = np.zeros((len(self.values), len(self.data)))
    
    def token_similarity(self) -> np.array:
            for i in range(len(self.values)):
                for j in range(len(self.data)):
                    self.similarities[i,j] = self.similarity_function(self.values[i].lower(), self.data[j].lower())
            return self.similarities
    
    def most_similar_attributes(self, algorithm: str, threshold: float) -> dict:
        self.similar_attributes = {}
        if algorithm == 'Cosine similarity':
            self.similarity_function = dst.cosine.similarity
            similarity = self.token_similarity()
        elif algorithm == 'Jaccard index':
            self.similarity_function = dst.jaccard.similarity
            similarity = self.token_similarity()
        elif algorithm == 'Sorensen–Dice coefficient':
            self.similarity_function = dst.sorensen.similarity
            similarity = self.token_similarity()
        for i in range(len(self.values)):
            max_value = max(similarity[i])
            if max_value >= threshold:
                max_index = np.argmax(similarity[i])
                self.similar_attributes[self.values[i]] = self.data[max_index]
        return self.similar_attributes
    

    def normalise_attributes(self, algorithm:str,  threshold: float) -> pd.DataFrame:
        similar_attributes = self.most_similar_attributes(algorithm, threshold)
        print(similar_attributes)
        list_of_lists = []
        for index, row in self.df.iterrows():
            try:
                new_attr = similar_attributes[row['Attribute']]
            except:
               new_attr = row['Attribute']
            list_of_lists.append([new_attr, row['Value'], row['Value']])
        return pd.DataFrame(list_of_lists, columns=['Attribute', 'Value', 'dummy'])
