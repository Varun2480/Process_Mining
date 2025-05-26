import pandas as pd
import random
from datetime import datetime, timedelta

def generate_mock_incidents(num_incidents=100):
    """Generates a DataFrame with mock incident data."""
    incident_types = [
        "Network connectivity issue",
        "Database performance degradation",
        "User authentication failure",
        "Application crash on login",
        "Security breach attempt",
        "Email delivery delay",
        "Storage capacity alert",
        "API endpoint timeout",
        "Server CPU spike",
        "Printer not responding"
    ]

    incidents_data = []
    for i in range(num_incidents):
        incident_type = random.choice(incident_types)
        description_template = random.choice([
            "Frequent reports of {type} for users in {region}.",
            "Investigating {type}. Affecting {department} users.",
            "Alert triggered: {type} detected on {system_name}.",
            "Troubleshooting {type}. Root cause unknown.",
            "Resolved: {type} after {solution_step}.",
            "Urgent: {type} impacts {service_name} availability."
        ])
        region = random.choice(["EMEA", "APAC", "AMER", "Global"])
        department = random.choice(["Sales", "Marketing", "IT", "HR", "Finance"])
        system_name = random.choice(["Prod-DB-01", "Web-Server-02", "Auth-Service", "Network-Switch-HQ"])
        service_name = random.choice(["CRM", "ERP", "Email Service", "Internal Portal"])
        solution_step = random.choice([
            "restarting service", "applying patch", "clearing cache",
            "scaling up resources", "checking firewall rules"
        ])

        description = description_template.format(
            type=incident_type, region=region, department=department,
            system_name=system_name, service_name=service_name, solution_step=solution_step
        )

        created_at = datetime.now() - timedelta(days=random.randint(1, 365), hours=random.randint(1, 24))
        updated_at = created_at + timedelta(hours=random.randint(1, 48))

        incidents_data.append({
            'id': f'INC-{1000 + i}',
            'description': description,
            'created_at': created_at,
            'updated_at': updated_at,
            'status': random.choice(['Open', 'In Progress', 'Resolved', 'Closed']),
            'priority': random.choice(['High', 'Medium', 'Low'])
        })

    return pd.DataFrame(incidents_data)

# Generate n mock incidents
number_of_incidents = 10
incidents_df = generate_mock_incidents(number_of_incidents)
print(f"Generated {len(incidents_df)} mock incidents.")
print(incidents_df.head())


# Write the DataFrame to a CSV file
csv_file_path = f"datasets/mock_incidents_{number_of_incidents}.csv"
incidents_df.to_csv(csv_file_path, index=False)
print(f"Mock incidents data saved to {csv_file_path}")