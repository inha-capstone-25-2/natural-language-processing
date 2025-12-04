from worker import process_paper
import time

print("Sending test task...")
# 테스트용 더미 데이터
paper_id = "test_paper_001"
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans.
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
"""

# 비동기 작업 전송
task = process_paper.delay(paper_id=paper_id, text=text)
print(f"Task sent! ID: {task.id}")

# 결과 대기 (실제로는 DB를 확인해야 하지만 여기서는 Celery 결과를 기다림)
# 주의: backend가 설정되어 있어야 결과를 받을 수 있음
try:
    result = task.get(timeout=300) # 5분 대기
    print(f"Task result: {result}")
except Exception as e:
    print(f"Error or timeout: {e}")
