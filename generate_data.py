import csv
import os
import random
from datetime import date, timedelta

# Headers based on constants.py and database_utils.py
# The database creation script will clean these names (e.g., "VENDOR NAME 1" -> "VENDOR_NAME_1")
headers = ["VENDOR NAME 1", "ITEM TOTAL COST", "ITEM DESCRIPTION", "COMMODITY DESCRIPTION", "DATE COLUMN"]

# Sample data
vendors = ["Dell Inc", "IBM Corp", "Microsoft", "Oracle Systems", "HP Enterprise", "Amazon Web Services", "Google LLC"]
commodities = ["IT Hardware", "Software Licensing", "Cloud Services", "Consulting Services", "Office Supplies"]
descriptions = [
    "Latitude 7420 Laptop", "Db2 License Renewal", "Azure Cloud Hosting", "Database Migration Consulting", "Box of A4 Paper",
    "PowerEdge R750 Server", "Windows 11 Enterprise License", "AWS S3 Storage", "Security Audit", "Ergonomic Office Chairs"
]

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Generate CSV data
with open('data/temp_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    today = date.today()
    for i in range(50): # Generate 50 rows of data
        vendor = random.choice(vendors)
        cost = round(random.uniform(100.0, 50000.0), 2)
        description = random.choice(descriptions)
        commodity = random.choice(commodities)
        # Generate dates over the last year
        random_date = today - timedelta(days=random.randint(0, 365))

        writer.writerow([vendor, cost, description, commodity, random_date.strftime('%Y-%m-%d')])

print("Successfully created data/temp_data.csv")
