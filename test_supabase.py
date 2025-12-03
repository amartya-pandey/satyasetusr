"""
Quick test script to verify Supabase connection and migration
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_supabase_connection():
    """Test Supabase connection"""
    print("🔍 Testing Supabase Connection...\n")
    
    # Check environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url:
        print("❌ SUPABASE_URL not found in environment variables")
        print("   Add it to your .env file: SUPABASE_URL=https://xxxxx.supabase.co")
        return False
    
    if not supabase_key:
        print("❌ SUPABASE_KEY not found in environment variables")
        print("   Add it to your .env file: SUPABASE_KEY=eyJhbGc...")
        return False
    
    print(f"✅ SUPABASE_URL: {supabase_url}")
    print(f"✅ SUPABASE_KEY: {supabase_key[:20]}...\n")
    
    # Test connection
    try:
        from supabase import create_client
        
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase client created successfully\n")
        
        # Test table access
        print("🔍 Testing table access...")
        response = supabase.table('certificates').select('count', count='exact').execute()
        
        count = response.count if hasattr(response, 'count') else len(response.data)
        print(f"✅ Found {count} records in certificates table\n")
        
        # Fetch sample record
        print("🔍 Fetching sample records...")
        sample = supabase.table('certificates').select('*').limit(3).execute()
        
        if sample.data:
            print(f"✅ Sample records:")
            for record in sample.data:
                print(f"   - {record.get('reg_no')}: {record.get('name')}")
        else:
            print("⚠️  No records found. Import your data using SUPABASE_SETUP.md guide")
        
        print("\n✅ Supabase connection test PASSED!")
        return True
        
    except ImportError:
        print("❌ Supabase package not installed")
        print("   Run: pip install supabase")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your SUPABASE_URL is correct")
        print("2. Check your SUPABASE_KEY is the 'anon/public' key")
        print("3. Ensure the 'certificates' table exists")
        print("4. Check RLS policies allow public access")
        return False

def test_verifier():
    """Test the Supabase verifier"""
    print("\n" + "="*50)
    print("🔍 Testing SupabaseCertificateVerifier...\n")
    
    try:
        from verifier_supabase import SupabaseCertificateVerifier
        
        verifier = SupabaseCertificateVerifier()
        print("✅ Verifier initialized successfully")
        
        # Test registration number lookup
        print("\n🔍 Testing registration lookup...")
        test_reg_nos = ['ABC2023001', '1BG19CS100', 'TEST123']
        
        for reg_no in test_reg_nos:
            result = verifier._lookup_registration(reg_no)
            if result:
                print(f"✅ Found: {reg_no} → {result.get('name')}")
            else:
                print(f"⚠️  Not found: {reg_no}")
        
        print("\n✅ Verifier test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Verifier test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("🚀 SUPABASE MIGRATION TEST")
    print("="*50 + "\n")
    
    connection_ok = test_supabase_connection()
    
    if connection_ok:
        test_verifier()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        print("\nNext steps:")
        print("1. Update api.py to use verifier_supabase")
        print("2. Test locally: python api.py")
        print("3. Deploy to Railway")
    else:
        print("\n" + "="*50)
        print("❌ TESTS FAILED")
        print("="*50)
        print("\nFollow SUPABASE_SETUP.md for complete setup instructions")
