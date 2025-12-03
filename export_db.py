"""Export SQLite database to SQL format for Supabase migration"""
import sqlite3

conn = sqlite3.connect('certs.db')

# Export to SQL file
with open('db_export.sql', 'w', encoding='utf-8') as f:
    for line in conn.iterdump():
        f.write(f'{line}\n')

print('✅ Exported to db_export.sql')

# Also create CSV export
import csv
cursor = conn.cursor()
cursor.execute("SELECT * FROM certificates")
rows = cursor.fetchall()
columns = [desc[0] for desc in cursor.description]

with open('certificates_export.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)

print(f'✅ Exported {len(rows)} records to certificates_export.csv')
print(f'Columns: {", ".join(columns)}')

conn.close()
