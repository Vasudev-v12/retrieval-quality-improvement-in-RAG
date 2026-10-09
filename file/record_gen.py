import json
import random
from datetime import datetime

# Setup seed for reproducibility
random.seed(42)

def make_record():
    first_names = ["Arjun", "Priya", "Rahul", "Ananya", "Rohan", "Sneha", "Vikram", "Neha", "Amit", "Divya",
                "John", "Sarah", "Michael", "Emma", "David", "Olivia", "James", "Sophia", "Robert", "Isabella"]
    last_names = ["Sharma", "Verma", "Kumar", "Singh", "Patel", "Joshi", "Nair", "Das", "Reddy", "Gupta",
                "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]

    # Generate 50 unique base employee templates
    employees_base = []
    for i in range(101, 151):
        emp_id = f"EMP{i}"
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        base_salary = round(random.uniform(50000, 140000), 2)
        department = random.choice(["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"])
        role = random.choice(["Junior Associate", "Senior Specialist", "Manager", "Lead Consultant", "Director"])
        employees_base.append({
            "emp_id": emp_id,
            "name": name,
            "base_salary": base_salary,
            "department": department,
            "role": role
        })

    months_2025 = [
        ("January", "2025-01-31"), ("February", "2025-02-28"), ("March", "2025-03-31"),
        ("April", "2025-04-30"), ("May", "2025-05-31"), ("June", "2025-06-30"),
        ("July", "2025-07-31"), ("August", "2025-08-31"), ("September", "2025-09-30"),
        ("October", "2025-10-31"), ("November", "2025-11-30"), ("December", "2025-12-31")
    ]

    records = []
    # Generate 200 monthly performance/payout records
    for _ in range(200):
        base_emp = random.choice(employees_base)
        month_name, date_str = random.choice(months_2025)
        
        shift_hours_worked = round(random.uniform(150, 190), 1)
        # 30% chance of getting a performance bonus in any given month
        bonus = round(random.uniform(500, 5000), 2) if random.random() < 0.3 else 0.0
        overtime_hours = max(0.0, round(shift_hours_worked - 168.0, 1))
        
        # Complex metadata string summarizing the monthly state
        performance_rating = random.choice(["Exceeds Expectations", "Meets Expectations", "Needs Improvement"])
        project_code = f"PRJ-{random.randint(400, 499)}"
        
        record_str = (
            f"Employee ID: {base_emp['emp_id']} | "
            f"Name: {base_emp['name']} | "
            f"Department: {base_emp['department']} | "
            f"Role: {base_emp['role']} | "
            f"Base Annual Salary: ${base_emp['base_salary']:,} | "
            f"Month: {month_name} | "
            f"Record Date: {date_str} | "
            f"Total Shift Hours Worked: {shift_hours_worked} hrs | "
            f"Overtime Hours: {overtime_hours} hrs | "
            f"Monthly Bonus Received: ${bonus:,} | "
            f"Performance Review: {performance_rating} | "
            f"Assigned Project Code: {project_code}"
        )
        records.append(record_str)

    # Show structure and some records
    print(f"Total records generated: {len(records)}")

    for r in records:
        print(r)
    return records
