import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timetable_generatorneww  # Import the timetable generator script

# Load the trained model
model_path = "best_admission_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Debugging step
print(f"Model loaded successfully! Type: {type(model)}")

# Load dataset
data_path = "updated_cleaned_admission_dataset.csv"
df = pd.read_csv(data_path)

# Streamlit App
st.set_page_config(page_title="College Admission Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Dashboard", "Predict Admissions", "Resource Planning", "Operational Tools"]
page = st.sidebar.radio("Go to", pages)

# Dashboard
if page == "Dashboard":
    st.title("📊 Admission Prediction Dashboard")
    st.markdown("Analyze past admission trends and forecast future enrollments.")

    # Admission trend over the years
    st.subheader("Yearly Admission Trends")
    admission_trends = df.groupby("Year")["Total_Applicants"].sum().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=admission_trends, x="Year", y="Total_Applicants", marker="o", ax=ax)
    st.pyplot(fig)

    # Competition Ratio Analysis
    st.subheader("Admission Competition Ratio")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Admission_Competition_Ratio"], kde=True, ax=ax)
    st.pyplot(fig)

elif page == "Predict Admissions":
    st.title("🎯 Admission Prediction")

    # Input fields (matching dataset)
    st.subheader("Enter Student Details")
    high_school_percentage = st.number_input("High School Percentage", min_value=0.0, max_value=100.0, step=0.1)
    entrance_exam_score = st.number_input("Entrance Exam Score", min_value=0, max_value=300, step=1)
    cgpa_1st_year = st.number_input("CGPA 1st Year", min_value=0.0, max_value=10.0, step=0.1)
    cgpa_2nd_year = st.number_input("CGPA 2nd Year", min_value=0.0, max_value=10.0, step=0.1)
    cgpa_3rd_year = st.number_input("CGPA 3rd Year", min_value=0.0, max_value=10.0, step=0.1)
    cgpa_4th_year = st.number_input("CGPA 4th Year", min_value=0.0, max_value=10.0, step=0.1)
    placement_rate = st.number_input("Placement Rate", min_value=0.0, max_value=100.0, step=0.1)
    average_salary = st.number_input("Average Salary", min_value=0.0, max_value=1000000.0, step=1000.0)
    economic_condition_index = st.slider("Economic Condition Index", min_value=1.0, max_value=10.0, step=0.1)
    total_applicants = st.number_input("Total Applicants", min_value=0, step=1)
    seats_available = st.number_input("Seats Available", min_value=1, step=1)

    # Predict Button
    if st.button("Predict Admission Probability"):
        input_data = np.array([[high_school_percentage, entrance_exam_score, cgpa_1st_year, cgpa_2nd_year,
                                cgpa_3rd_year, cgpa_4th_year, placement_rate, average_salary,
                                economic_condition_index, total_applicants, seats_available]])

        # Predict
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Admission Probability: {prediction:.2f}%")


# Resource Planning
elif page == "Resource Planning":
    st.title("📅 Resource Allocation Planning")
    st.markdown("Plan faculty hiring, classroom allocation, and budget based on predicted enrollments.")

    year = st.selectbox("Select Year", sorted(df['Year'].unique()))
    selected_year_data = df[df['Year'] == year]
    total_students = selected_year_data['Seats_Available'].sum()

    # Define allocation rules
    def calculate_faculty_needed(total_students, faculty_ratio=20):
        return total_students // faculty_ratio

    def allocate_classrooms_and_labs(total_students, classroom_capacity=40, lab_capacity=30):
        classrooms_needed = total_students // classroom_capacity
        labs_needed = total_students // lab_capacity
        return classrooms_needed, labs_needed

    faculty_needed = calculate_faculty_needed(total_students)
    classrooms_needed, labs_needed = allocate_classrooms_and_labs(total_students)

    st.subheader(f"Resource Requirements for {year}")
    st.write(f"📌 Estimated Faculty Required: {faculty_needed}")
    st.write(f"🏫 Estimated Classrooms Required: {classrooms_needed}")
    st.write(f"🔬 Estimated Labs Required: {labs_needed}")

    # Budget Estimation
    def estimate_budget(faculty_count, avg_salary, total_applicants, marketing_cost_per_applicant, classrooms, labs, classroom_cost, lab_cost):
        faculty_salary_budget = faculty_count * avg_salary
        marketing_budget = total_applicants * marketing_cost_per_applicant
        infrastructure_budget = (classrooms * classroom_cost) + (labs * lab_cost)
        return faculty_salary_budget, marketing_budget, infrastructure_budget

    avg_salary = df['Average_Salary'].mean()
    faculty_budget, marketing_budget, infra_budget = estimate_budget(
        faculty_needed, avg_salary, total_students * 2, 50, classrooms_needed, labs_needed, 50000, 100000
    )

    st.subheader("💰 Budget Estimation")
    st.write(f"👨‍🏫 Faculty Salary Budget: ₹{faculty_budget:,.2f}")
    st.write(f"📢 Suggested Marketing Budget: ₹{marketing_budget:,.2f}")
    st.write(f"🏗️ Infrastructure Budget: ₹{infra_budget:,.2f}")

# Operational Tools
elif page == "Operational Tools":
    st.title("⚙️ Operational Efficiency Tools")

    # Timetable Generation
    st.subheader("🕒 Timetable Generation")

    # Run timetable generator script
    timetable_path = "advanced_timetable.csv"

    # Load generated timetable
    try:
        df_timetable = pd.read_csv(timetable_path)
        st.dataframe(df_timetable)

        # Download option
        csv = df_timetable.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Timetable CSV", csv, "timetable.csv", "text/csv", key="download-csv")

    except FileNotFoundError:
        st.error("Timetable file not found. Generating a new timetable...")

        # Define days and time slots
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        time_slots = ["9:00-10:00", "10:00-11:00", "11:15-12:15", "1:15-2:15", "2:30-3:30"]

        # Define courses and their subjects
        courses = {
            "Computer Science": ["Data Structures", "Algorithms", "AI & ML", "Cybersecurity", "Operating Systems"],
            "Mechanical Engineering": ["Thermodynamics", "Fluid Mechanics", "Machine Design", "Robotics", "Manufacturing"],
            "Electronics": ["Circuit Analysis", "Microcontrollers", "VLSI Design", "Digital Signal Processing", "Embedded Systems"],
            "AI & Data Science": ["Deep Learning", "NLP", "Big Data Analytics", "Cloud Computing", "Data Visualization"],
        }

        # Sample faculty members
        faculty_members = ["Dr. Smith", "Prof. Johnson", "Dr. Lee", "Dr. Brown", "Prof. Williams"]

        # Sample classrooms & labs
        classrooms = ["Room A1", "Room B2", "Room C3", "Lab D1", "Lab E2"]

        # Generate timetable
        timetable = []
        for course, subjects in courses.items():
            for day in days:
                for slot in time_slots:
                    subject = np.random.choice(subjects)
                    faculty = np.random.choice(faculty_members)
                    location = np.random.choice(classrooms)
                    timetable.append([course, day, slot, subject, faculty, location])

        # Convert to DataFrame
        df_timetable = pd.DataFrame(timetable, columns=["Course", "Day", "Time Slot", "Subject", "Faculty", "Location"])

        # Display timetable
        st.dataframe(df_timetable)

        # Export timetable as CSV
        csv = df_timetable.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Timetable CSV", csv, "timetable.csv", "text/csv", key="download-csv")

st.sidebar.markdown("Developed by MHSSCE | AIML03")
