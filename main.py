import json
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

class ToDoList:
    def __init__(self):
        self.tasks = []
        self.load_tasks()

    def add_task(self, task):
        """Add a new task to the to-do list and save it."""
        self.tasks.append({
            'task': task,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.save_tasks()

    def load_tasks(self):
        """Load tasks from a JSON file."""
        try:
            with open('tasks.json', 'r') as file:
                self.tasks = json.load(file)
        except FileNotFoundError:
            self.tasks = []

    def save_tasks(self):
        """Save tasks to a JSON file."""
        with open('tasks.json', 'w') as file:
            json.dump(self.tasks, file)

    def get_recommendations(self, task):
        """Get task recommendations based on the current task using machine learning."""
        if not self.tasks:
            return []

        df = pd.DataFrame(self.tasks)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['task'])

        new_task_vec = vectorizer.transform([task])
        nn = NearestNeighbors(n_neighbors=3).fit(X)
        distances, indices = nn.kneighbors(new_task_vec)

        recommendations = df.iloc[indices[0]]['task'].tolist()
        return recommendations

if __name__ == "__main__":
    todo = ToDoList()
    while True:
        command = input("Enter a command (add, recommend, quit): ").strip().lower()
        if command == "add":
            task = input("Enter the task: ")
            todo.add_task(task)
            print("Task added.")
        elif command == "recommend":
            task = input("Enter the task to get recommendations for: ")
            recommendations = todo.get_recommendations(task)
            if recommendations:
                print("Recommended tasks:")
                for rec in recommendations:
                    print(f"- {rec}")
            else:
                print("No recommendations available.")
        elif command == "quit":
            break
        else:
            print("Invalid command.")
