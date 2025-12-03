# 🚀 Supabase Migration Guide

## Step 1: Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Click **"Start your project"** or **"New Project"**
3. Create a new project:
   - **Name**: certificate-verifier-db (or any name)
   - **Database Password**: (save this securely)
   - **Region**: Choose closest to you
4. Wait for project to provision (~2 minutes)

## Step 2: Create Database Table

1. In Supabase Dashboard, go to **SQL Editor**
2. Click **"New Query"**
3. Paste this SQL and click **"Run"**:

```sql
CREATE TABLE certificates (
  id BIGSERIAL PRIMARY KEY,
  reg_no TEXT UNIQUE,
  name TEXT NOT NULL,
  institution TEXT,
  degree TEXT,
  year INTEGER,
  notes TEXT,
  father_name TEXT,
  usn TEXT,
  assigned_date DATE,
  certificate_type TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster lookups
CREATE INDEX idx_reg_no ON certificates(reg_no);
CREATE INDEX idx_usn ON certificates(usn);
CREATE INDEX idx_name ON certificates(name);

-- Enable Row Level Security (RLS)
ALTER TABLE certificates ENABLE ROW LEVEL SECURITY;

-- Create policy for public read access (anyone can read)
CREATE POLICY "Public read access"
  ON certificates
  FOR SELECT
  TO public
  USING (true);

-- Create policy for authenticated write access (optional - can make public too)
CREATE POLICY "Authenticated write access"
  ON certificates
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated update access"
  ON certificates
  FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);
```

## Step 3: Import Your Data

### Option A: Using CSV (Easiest)
1. Go to **Table Editor** → **certificates** table
2. Click **"Insert"** → **"Import data from CSV"**
3. Upload the file: `certificates_export.csv`
4. Map columns and click **"Import"**

### Option B: Using SQL
1. Open `db_export.sql` in a text editor
2. Copy the INSERT statements
3. Paste in SQL Editor and run

## Step 4: Get Your Credentials

1. Go to **Settings** → **API**
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon/public key**: `eyJhbGc...` (long key)

## Step 5: Configure Your Application

### For Local Development:
Create/update `.env` file:
```env
# OCR API Key
OCRSPACE_API_KEY=your_ocr_key_here

# Supabase Credentials
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### For Railway Deployment:
1. Go to Railway project **Variables**
2. Add:
   - `SUPABASE_URL` = `https://xxxxx.supabase.co`
   - `SUPABASE_KEY` = `eyJhbGc...`
   - Keep existing `OCRSPACE_API_KEY`

## Step 6: Update Your Code

Replace in `api.py`:
```python
# OLD:
from verifier import CertificateVerifier
verifier = CertificateVerifier()

# NEW:
from verifier_supabase import SupabaseCertificateVerifier
verifier = SupabaseCertificateVerifier()
```

## Step 7: Test Locally

```bash
# Install new dependency
pip install supabase

# Test the API
python -c "from verifier_supabase import SupabaseCertificateVerifier; v = SupabaseCertificateVerifier(); print('✅ Connected to Supabase!')"

# Start server
python api.py
```

## Step 8: Deploy to Railway

```bash
git add .
git commit -m "feat: Migrate to Supabase cloud database"
git push amartya_repo main
```

Railway will:
1. Detect new requirements (supabase)
2. Read environment variables
3. Auto-deploy with cloud database

---

## 🎉 Benefits of Supabase:

✅ **Public Access** - Anyone can read/write to your database  
✅ **Real-time** - Changes sync instantly  
✅ **Dashboard** - Easy web UI to manage data  
✅ **Backups** - Automatic database backups  
✅ **Scalable** - Handles thousands of requests  
✅ **Free Tier** - 500MB database, unlimited API calls  

## 📊 Managing Data:

### Add New Certificate via Dashboard:
1. Go to Table Editor → certificates
2. Click "Insert row"
3. Fill in details
4. Save

### Add via API (Python):
```python
from supabase import create_client

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

supabase.table('certificates').insert({
    "reg_no": "ABC2023999",
    "name": "New Student",
    "father_name": "Father Name",
    "institution": "University Name",
    "degree": "BCA",
    "year": 2023
}).execute()
```

## 🔒 Security Notes:

- The **anon key** is safe to expose (it's for public access)
- Row Level Security (RLS) policies control who can do what
- Current setup: Anyone can READ, only authenticated can WRITE
- To allow public write, modify policies in SQL Editor

---

## Need Help?

- Supabase Docs: https://supabase.com/docs
- Your project dashboard: Check Supabase.com → Your Projects
- Test connection: Use the provided Python test script
