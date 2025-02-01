import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


class RecommendationClass:
    def __init__(self,data_path):

        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)

        #Dropping course url columns
        drop_cols = ["Course URL"]
        self.df.drop(columns=drop_cols, inplace=True)

        #Dropping duplicate rows
        self.df.drop_duplicates(inplace=True)

        #self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        #self.indices = pd.Series(df.index,index=df['Course Name'])

        self.titles = self.df['Course Name']
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.titles)

    def get_recommendations(self,input_string):
        input_vector = self.tfidf_vectorizer.transform([input_string])
        cosine_similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
        
        # Sort courses based on similarity scores
        sim_scores = sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[:10]  # Top 10 results
        
        # Extract course titles and similarity scores
        course_indices = [i[0] for i in sim_scores]
        similarity_scores = [round(i[1], 3) for i in sim_scores]
        #Return course name, desc, uni, rating, difficulty level and sim score
        recommendations = list(zip(self.titles.iloc[course_indices],
                                    self.df['University'].iloc[course_indices],
                                    self.df['Course Rating'].iloc[course_indices],
                                    self.df['Course Description'].iloc[course_indices],
                                    self.df['Difficulty Level'].iloc[course_indices],
                                    similarity_scores))
        return recommendations