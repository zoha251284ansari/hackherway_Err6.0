import pickle
import numpy as np

# Load the trained model
model_path = "admission_modelneww.pkl"  # Update with actual model path
with open(model_path, "rb") as file:
    model = pickle.load(file)

def predict_admission(hs_percentage, exam_score, economic_index, total_applicants, seats_available):
    input_data = np.array([[hs_percentage, exam_score, economic_index, total_applicants, seats_available]])
    prediction = model.predict(input_data)[0]
    return prediction

if __name__ == "__main__":
    print("Enter student details for admission prediction:")
    hs_percentage = float(input("High School Percentage: "))
    exam_score = int(input("Entrance Exam Score: "))
    economic_index = float(input("Economic Condition Index (1-10): "))
    total_applicants = int(input("Total Applicants: "))
    seats_available = int(input("Seats Available: "))

    prediction = predict_admission(hs_percentage, exam_score, economic_index, total_applicants, seats_available)
    print(f"Predicted Admission Probability: {prediction:.2f}%")
