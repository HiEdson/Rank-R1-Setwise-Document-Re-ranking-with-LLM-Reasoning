import json
import os

def generate_data():
    os.makedirs('sample_data', exist_ok=True)

    # 1. Mock Queries
    queries = {
        "q1": "What is the capital of France?",
        "q2": "How to cook a perfect steak?"
    }
    with open('sample_data/queries.json', 'w') as f:
        json.dump(queries, f, indent=4)

    # 2. Mock Documents
    docs = {
        "d1": {"title": "Paris Guide", "text": "Paris is the capital and most populous city of France."},
        "d2": {"title": "London Info", "text": "London is the capital of England and the United Kingdom."},
        "d3": {"title": "Steak Recipe", "text": "To cook a perfect steak, use a cast iron skillet and plenty of butter."},
        "d4": {"title": "Vegetable Soup", "text": "This soup is made with carrots, celery, and onions."},
        "d5": {"title": "France Trivia", "text": "France is a country in Western Europe."}
    }
    with open('sample_data/docs.json', 'w') as f:
        json.dump(docs, f, indent=4)

    # 3. Simulated BM25 Run (TREC Format: qid Q0 docid rank score tag)
    # Goal: Simulated a bad initial ranking to see if LLM improves it.
    run_lines = [
        "q1 Q0 d2 1 10.5 BM25",  # London (Wrong)
        "q1 Q0 d5 2 9.2 BM25",   # France Trivia (Weak)
        "q1 Q0 d1 3 8.1 BM25",   # Paris (Correct, but ranked 3rd)
        "q2 Q0 d4 1 12.0 BM25",  # Soup (Wrong)
        "q2 Q0 d3 2 11.5 BM25"   # Steak (Correct, but ranked 2nd)
    ]
    with open('sample_data/bm25_simulated.run', 'w') as f:
        f.write("\n".join(run_lines) + "\n")

    print("Simulated data generated in sample_data/ directory.")

if __name__ == "__main__":
    generate_data()
