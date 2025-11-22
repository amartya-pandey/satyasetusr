import sqlite3
import os

def init_database():
    """Initialize the SQLite database with sample certificate records."""
    
    db_path = "certs.db"
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Create connection and table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS certificates(
        reg_no TEXT PRIMARY KEY,
        name TEXT,
        institution TEXT,
        degree TEXT,
        year INTEGER,
        notes TEXT
    )
    ''')
    
    # Sample data
    sample_data = [
        ('ABC2023001', 'Saksham Sharma', 'DevLabs Institute', 'B.Tech Computer Engg', 2023, 'Sample student'),
        ('ABC2022007', 'Prisha Verma', 'Global Tech University', 'M.Tech AI', 2022, 'Sample student'),
        ('UNI10009', 'Rajeev Kumar', 'Northfield University', 'B.Sc Physics', 2019, 'Sample student'),
        ('INSTX-555', 'Anita Desai', 'Sunrise Polytechnic', 'Diploma Civil', 2021, 'Sample student'),
        ('COLL-7788', 'John Doe', 'WestEnd College', 'BBA', 2020, 'Sample placeholder'),
        ('CERT-9001', 'Maya Iyer', 'Metro University', 'MSc Maths', 2018, 'Sample student'),
        ('REG-2021-345', 'Ram Singh', 'City College', 'BCom', 2021, 'Sample student'),
        ('EDU-3333', 'Nina Gupta', 'Coastal Institute', 'BCA', 2025, 'Sample student'),
        ('COL-1212', 'Alex Wong', 'Global Tech University', 'B.Tech ECE', 2022, 'Sample student'),
        ('STU-0007', 'Liu Chen', 'International Academy', 'PhD Chemistry', 2020, 'Sample student')
    ]
    
    # Insert sample data
    cursor.executemany('''
    INSERT OR IGNORE INTO certificates 
    (reg_no, name, institution, degree, year, notes) 
    VALUES (?, ?, ?, ?, ?, ?)
    ''', sample_data)
    
    conn.commit()
    print(f"Database initialized with {cursor.rowcount} records")
    
    # Verify data
    cursor.execute("SELECT COUNT(*) FROM certificates")
    count = cursor.fetchone()[0]
    print(f"Total records in database: {count}")
    
    # Show sample records
    cursor.execute("SELECT * FROM certificates LIMIT 3")
    sample_records = cursor.fetchall()
    print("\nSample records:")
    for record in sample_records:
        print(f"  {record}")
    
    conn.close()
    print(f"\nDatabase '{db_path}' created successfully!")

if __name__ == "__main__":
    init_database()
