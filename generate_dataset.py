import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

num_samples = 2500

# Degrees and their probabilities
degrees = ['B.Tech', 'MCA', 'BCA', 'B.Sc', 'B.Com', 'BBA', 'MBA']
degree_probs = [0.35, 0.10, 0.10, 0.10, 0.15, 0.10, 0.10]
selected_degrees = np.random.choice(degrees, size=num_samples, p=degree_probs)

# Streams mapping
streams_map = {
    'B.Tech': ['Computer Science', 'Information Technology', 'Electronics', 'Mechanical', 'Civil'],
    'MCA': ['Computer Applications'],
    'BCA': ['Computer Applications'],
    'B.Sc': ['Computer Science', 'Mathematics', 'Physics', 'Biotech'],
    'B.Com': ['Accounting & Finance', 'General Commerce'],
    'BBA': ['Marketing', 'Finance', 'Human Resources', 'General Management'],
    'MBA': ['Marketing', 'Finance', 'Human Resources', 'Operations', 'Business Analytics']
}

# Stream probabilities within degrees
streams_probs = {
    'B.Tech': [0.4, 0.25, 0.15, 0.1, 0.1],
    'MCA': [1.0],
    'BCA': [1.0],
    'B.Sc': [0.4, 0.2, 0.2, 0.2],
    'B.Com': [0.6, 0.4],
    'BBA': [0.3, 0.3, 0.2, 0.2],
    'MBA': [0.25, 0.25, 0.2, 0.15, 0.15]
}

selected_streams = []
for deg in selected_degrees:
    opts = streams_map[deg]
    probs = streams_probs[deg]
    selected_streams.append(np.random.choice(opts, p=probs))

# Basic demographics & academic performance
gender = np.random.choice(['M', 'F'], size=num_samples, p=[0.6, 0.4])
ssc_p = np.random.normal(70, 10, num_samples).clip(45, 99)
hsc_p = np.random.normal(70, 10, num_samples).clip(45, 99)
cgpa = np.random.normal(7.5, 1.2, num_samples).clip(4.0, 10.0)

# Work experience
workex = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.18, 0.82])

# Skills (1-10 scale)
coding_skills = np.zeros(num_samples)
communication_skills = np.random.normal(6.5, 1.5, num_samples).clip(1, 10)
analytical_skills = np.random.normal(6.5, 1.5, num_samples).clip(1, 10)
domain_knowledge = np.random.normal(6.5, 1.5, num_samples).clip(1, 10)

# Projects, Internships, Certifications
projects = np.zeros(num_samples, dtype=int)
internships = np.zeros(num_samples, dtype=int)
certifications = np.zeros(num_samples, dtype=int)

# Setup specialized attributes based on degree/stream
for i in range(num_samples):
    deg = selected_degrees[i]
    strm = selected_streams[i]
    
    # Coding skills (Tech degrees get higher baseline coding skills)
    if deg in ['B.Tech', 'MCA', 'BCA'] and strm in ['Computer Science', 'Information Technology', 'Computer Applications']:
        coding_skills[i] = np.clip(np.random.normal(7.5, 1.2), 3, 10)
        projects[i] = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
        internships[i] = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        certifications[i] = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
    elif deg == 'B.Tech' and strm == 'Electronics':
        coding_skills[i] = np.clip(np.random.normal(5.5, 1.5), 1, 10)
        projects[i] = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        internships[i] = np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
        certifications[i] = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
    elif deg == 'B.Sc' and strm == 'Computer Science':
        coding_skills[i] = np.clip(np.random.normal(6.0, 1.5), 2, 10)
        projects[i] = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
        internships[i] = np.random.choice([0, 1], p=[0.7, 0.3])
        certifications[i] = np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
    else:
        # Non-tech / core degrees
        coding_skills[i] = np.clip(np.random.normal(3.0, 1.5), 1, 10)
        projects[i] = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        internships[i] = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        certifications[i] = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])

    # Domain Knowledge adjustments
    if deg in ['MBA', 'BBA', 'B.Com']:
        domain_knowledge[i] = np.clip(np.random.normal(7.2, 1.2), 3, 10)
        # B.Com/BBA/MBA finance/analytcs gets analytical boost
        if strm in ['Finance', 'Accounting & Finance', 'Business Analytics']:
            analytical_skills[i] = np.clip(np.random.normal(7.5, 1.1), 3, 10)
    elif deg == 'B.Tech' and strm in ['Mechanical', 'Civil']:
        domain_knowledge[i] = np.clip(np.random.normal(7.5, 1.1), 3, 10)

# Determine Placement logic
placed_probs = []
placed_sector = []
placed_role = []
salary = []

for i in range(num_samples):
    deg = selected_degrees[i]
    strm = selected_streams[i]
    cg = cgpa[i]
    cod = coding_skills[i]
    comm = communication_skills[i]
    anl = analytical_skills[i]
    dom = domain_knowledge[i]
    proj = projects[i]
    intern = internships[i]
    cert = certifications[i]
    we = 1.0 if workex[i] == 'Yes' else 0.0
    
    # Calculate score weights for different sectors
    # 1. Tech Sector suitability
    tech_suit = (cod * 0.4) + (anl * 0.25) + (cg * 0.15) + (proj * 0.1) + (intern * 0.1)
    
    # 2. Finance Sector suitability
    fin_suit = (anl * 0.35) + (dom * 0.25) + (comm * 0.2) + (we * 0.1) + (cert * 0.1)
    
    # 3. Consulting suitability
    cons_suit = (comm * 0.35) + (anl * 0.3) + (cg * 0.15) + (we * 0.1) + (proj * 0.1)
    
    # 4. Core Engineering suitability
    core_suit = (dom * 0.45) + (anl * 0.25) + (cg * 0.15) + (proj * 0.15) if deg == 'B.Tech' and strm in ['Mechanical', 'Civil', 'Electronics'] else 0.0
    
    # 5. Marketing/HR suitability
    mkt_hr_suit = (comm * 0.4) + (dom * 0.25) + (we * 0.15) + (intern * 0.1) + (cert * 0.1)

    # Determine primary placed candidate sector
    scores = {
        'Tech': tech_suit if deg in ['B.Tech', 'MCA', 'BCA', 'B.Sc'] else tech_suit * 0.5,
        'Finance': fin_suit if strm in ['Finance', 'Accounting & Finance', 'Mathematics'] or deg in ['MBA', 'B.Com'] else fin_suit * 0.4,
        'Consulting': cons_suit,
        'Core Engineering': core_suit,
        'Marketing & HR': mkt_hr_suit if strm in ['Marketing', 'Human Resources', 'General Commerce', 'General Management'] or deg in ['MBA', 'BBA', 'B.Com'] else mkt_hr_suit * 0.4
    }
    
    best_sector = max(scores, key=scores.get)
    best_score = scores[best_sector]
    
    # Academic check: very low grades reduces placement chance
    acad_penalty = 1.0
    if cg < 6.0:
        acad_penalty = 0.5
    if cg < 5.0:
        acad_penalty = 0.2
        
    # Placement probability
    # Base probability on the suitability score of their best fit sector
    prob = (best_score / 10.0) * acad_penalty
    
    # Add project/internship bonuses
    prob += (intern * 0.08) + (proj * 0.04) + (cert * 0.03)
    prob = np.clip(prob, 0.05, 0.98)
    
    is_placed = np.random.choice([1, 0], p=[prob, 1 - prob])
    
    if is_placed == 1:
        placed_sector.append(best_sector)
        
        # Sector specific role and salary
        if best_sector == 'Tech':
            roles = ['Software Engineer', 'Data Analyst', 'Web Developer', 'QA Engineer', 'Cloud Associate']
            role_probs = [0.5, 0.2, 0.15, 0.1, 0.05]
            role = np.random.choice(roles, p=role_probs)
            placed_role.append(role)
            
            # Tech salary: highly dependent on coding skill and CGPA
            base_sal = 3.5
            coding_mult = (cod - 3) * 1.5 if cod > 3 else 0
            cg_mult = (cg - 6) * 1.0 if cg > 6 else 0
            sal = base_sal + coding_mult + cg_mult + (intern * 1.0)
            salary.append(round(np.clip(sal, 3.6, 25.0), 2))
            
        elif best_sector == 'Finance':
            roles = ['Financial Analyst', 'Risk Associate', 'Investment Banking Analyst', 'Tax Consultant']
            role_probs = [0.4, 0.3, 0.1, 0.2]
            role = np.random.choice(roles, p=role_probs)
            placed_role.append(role)
            
            # Finance salary: analytical skill and certs
            base_sal = 3.0
            anl_mult = (anl - 4) * 1.2 if anl > 4 else 0
            cert_mult = cert * 1.2
            sal = base_sal + anl_mult + cert_mult + (we * 1.5)
            salary.append(round(np.clip(sal, 3.0, 18.0), 2))
            
        elif best_sector == 'Consulting':
            roles = ['Business Analyst', 'Strategy Consultant', 'Operations Analyst']
            role_probs = [0.5, 0.2, 0.3]
            role = np.random.choice(roles, p=role_probs)
            placed_role.append(role)
            
            # Consulting salary: comms and analytical skills
            base_sal = 4.0
            comm_mult = (comm - 5) * 1.0 if comm > 5 else 0
            anl_mult = (anl - 5) * 1.0 if anl > 5 else 0
            sal = base_sal + comm_mult + anl_mult + (we * 1.8)
            salary.append(round(np.clip(sal, 4.0, 16.0), 2))
            
        elif best_sector == 'Core Engineering':
            roles = ['Graduate Engineer Trainee', 'Design Engineer', 'Site Engineer']
            role_probs = [0.6, 0.25, 0.15]
            role = np.random.choice(roles, p=role_probs)
            placed_role.append(role)
            
            # Core salary
            base_sal = 3.0
            dom_mult = (dom - 5) * 0.8 if dom > 5 else 0
            sal = base_sal + dom_mult + (proj * 0.5)
            salary.append(round(np.clip(sal, 3.0, 12.0), 2))
            
        else: # Marketing & HR
            roles = ['Marketing Executive', 'Sales Associate', 'HR Recruiter', 'Brand Executive']
            role_probs = [0.35, 0.3, 0.25, 0.1]
            role = np.random.choice(roles, p=role_probs)
            placed_role.append(role)
            
            # Marketing / HR salary
            base_sal = 2.8
            comm_mult = (comm - 4) * 0.8 if comm > 4 else 0
            dom_mult = (dom - 4) * 0.5 if dom > 4 else 0
            sal = base_sal + comm_mult + dom_mult + (we * 1.0)
            salary.append(round(np.clip(sal, 2.5, 10.0), 2))
            
    else:
        placed_sector.append('Not Placed')
        placed_role.append('None')
        salary.append(np.nan)

# Create DataFrame
df_expanded = pd.DataFrame({
    'gender': gender,
    'degree': selected_degrees,
    'stream': selected_streams,
    'ssc_p': np.round(ssc_p, 2),
    'hsc_p': np.round(hsc_p, 2),
    'cgpa': np.round(cgpa, 2),
    'workex': workex,
    'coding_skills': np.round(coding_skills, 1),
    'communication_skills': np.round(communication_skills, 1),
    'analytical_skills': np.round(analytical_skills, 1),
    'domain_knowledge': np.round(domain_knowledge, 1),
    'projects': projects,
    'internships': internships,
    'certifications': certifications,
    'placed_status': ['Placed' if p != 'Not Placed' else 'Not Placed' for p in placed_sector],
    'placed_sector': placed_sector,
    'placed_role': placed_role,
    'salary_lpa': salary
})

# Write to file
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(_BASE_DIR, 'Placement_Data_Expanded.csv')
df_expanded.to_csv(output_file, index=False)

print(f"Generated {num_samples} rows and saved to {output_file}")
print("Sector distribution:")
print(df_expanded['placed_sector'].value_counts())
