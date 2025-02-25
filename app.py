from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# 데이터 로드
mentees = pd.read_csv("mentees.csv")
mentors = pd.read_csv("mentors_with_korean_name.csv")

# 텍스트 데이터 결합
mentees["combined"] = mentees["categories"] + ", " + mentees["styles"] + ", " + mentees["environment"]
mentors["combined"] = mentors["categories"] + ", " + mentors["styles"] + ", " + mentors["environment"]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
mentee_matrix = vectorizer.fit_transform(mentees["combined"])
mentor_matrix = vectorizer.transform(mentors["combined"])

# 코사인 유사도 계산
similarity_matrix = cosine_similarity(mentee_matrix, mentor_matrix)

# 추천 결과 저장
recommendations = {}
for i, mentee_id in enumerate(mentees["mentee_id"]):
    best_match_indices = similarity_matrix[i].argsort()[-3:][::-1]  # 상위 3명 추출
    recommended_mentors = mentors.iloc[best_match_indices]["mentor_id"].tolist()
    similarity_scores = similarity_matrix[i][best_match_indices].tolist()
    
    # (멘토, 유사도) 형태로 저장
    mentor_similarity_pairs = [
        {"mentor_id": mentor, "similarity": round(similarity, 4)}
        for mentor, similarity in zip(recommended_mentors, similarity_scores)
    ]
    
    recommendations[mentee_id] = mentor_similarity_pairs

# API 엔드포인트 (멘티 ID로 추천 멘토 반환)
@app.get("/recommend/{mentee_id}")
def get_recommendations(mentee_id: int):
    if mentee_id in recommendations:
        return {"mentee_id": mentee_id, "recommendations": recommendations[mentee_id]}
    return {"error": "해당 멘티 ID가 존재하지 않습니다."}

# 서버 실행: uvicorn app:app --reload
