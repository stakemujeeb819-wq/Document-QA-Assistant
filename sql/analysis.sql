-- 1. JOIN: each question with its source chunks and scores
SELECT i.question, i.prompt_version, s.score, s.chunk_text
FROM interactions i
JOIN sources s ON s.interaction_id = i.id
ORDER BY i.id, s.score DESC;

-- 2. GROUP BY: how many questions were asked per prompt version
SELECT prompt_version, COUNT(*) AS total_questions
FROM interactions
GROUP BY prompt_version;

-- 3. SUBQUERY: questions whose average source score is below the overall average
SELECT i.question, i.prompt_version
FROM interactions i
WHERE (
    SELECT AVG(s.score) FROM sources s WHERE s.interaction_id = i.id
) < (
    SELECT AVG(score) FROM sources
);