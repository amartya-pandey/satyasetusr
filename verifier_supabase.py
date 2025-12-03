"""
Supabase Database Verifier
Replaces SQLite with cloud-hosted Supabase PostgreSQL database
"""
import os
from typing import Dict, Any, List, Optional, Tuple
from rapidfuzz import fuzz
import re
from dotenv import load_dotenv

load_dotenv()

class SupabaseCertificateVerifier:
    """Certificate verification engine using Supabase cloud database."""
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        """
        Initialize the verifier with Supabase connection.
        
        Args:
            supabase_url: Supabase project URL (or from SUPABASE_URL env var)
            supabase_key: Supabase anon/public key (or from SUPABASE_KEY env var)
        """
        try:
            from supabase import create_client, Client
        except ImportError:
            raise ImportError("Please install supabase: pip install supabase")
        
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials not provided. Set SUPABASE_URL and SUPABASE_KEY environment variables.")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Configurable regex patterns for registration number extraction
        self.reg_patterns = [
            r'\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}',    # 1BG19CS100 (VTU USN format)
            r'USN:?\s*\d[A-Z]{2}\d{2}[A-Z]{2}\d{3}',  # USN: 1BG19CS100
            r'[A-Z]{2,4}[-_]?\d{4}[-_]?\d{3}',  # ABC-2023-001 or ABC2023001
            r'[A-Z]{3,5}[-_]?\d{2,6}',          # UNI10009, INSTX-555
            r'REG[-_]?\d{4}[-_]?\d{3}',         # REG-2021-345
            r'CERT[-_]?\d{4}',                  # CERT-9001
            r'EDU[-_]?\d{4}',                   # EDU-3333
            r'COL[-_]?\d{4}',                   # COL-1212
            r'STU[-_]?\d{4}',                   # STU-0007
            r'[A-Z]+[-_]?\d+[-_]?[A-Z]*'       # General pattern
        ]
        
        # Field weights for final score calculation
        self.field_weights = {
            'name': 0.35,
            'father_name': 0.25,  # Father's name is important for verification
            'institution': 0.2,
            'degree': 0.15,
            'year': 0.05
        }
        
        # Decision thresholds
        self.authentic_threshold = 0.75
        self.suspect_threshold = 0.4
    
    def verify_certificate(self, ocr_result: Dict[str, Any], 
                          image_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a certificate using OCR results and Supabase database lookup.
        
        Args:
            ocr_result: OCR result dictionary from ocr_client
            image_filename: Original image filename (optional)
            
        Returns:
            Structured verification result
        """
        if not ocr_result.get('success', False):
            return {
                'registration_no': None,
                'db_record': None,
                'ocr_extracted': {'raw_text': ocr_result.get('error', 'OCR failed')},
                'field_scores': {},
                'final_score': 0.0,
                'decision': 'NOT_FOUND',
                'reasons': ['OCR processing failed'],
                'confidence': 0.0
            }
        
        extracted_text = ocr_result.get('extracted_text', '')
        
        # Step 1: Extract registration number
        reg_numbers = self._extract_registration_numbers(extracted_text)
        
        if not reg_numbers:
            return {
                'registration_no': None,
                'db_record': None,
                'ocr_extracted': {
                    'raw_text': extracted_text,
                    'name': None,
                    'institution': None,
                    'degree': None,
                    'year': None
                },
                'field_scores': {},
                'final_score': 0.0,
                'decision': 'NOT_FOUND',
                'reasons': ['No registration number found in OCR text'],
                'confidence': 0.0
            }
        
        # Try each registration number until we find a match
        best_result = None
        best_score = 0.0
        
        for reg_no in reg_numbers:
            # Step 2: Supabase lookup
            db_record = self._lookup_registration(reg_no)
            
            if db_record:
                # Step 3: Extract fields from OCR text
                ocr_extracted = self._extract_fields_from_ocr(extracted_text, db_record)
                
                # Step 4: Compare fields and calculate scores
                field_scores = self._compare_fields(db_record, ocr_extracted)
                final_score = self._calculate_final_score(field_scores)
                
                # Step 5: Make decision
                decision, reasons = self._make_decision(final_score, field_scores, reg_no)
                
                result = {
                    'registration_no': reg_no,
                    'db_record': db_record,
                    'ocr_extracted': ocr_extracted,
                    'field_scores': field_scores,
                    'final_score': final_score,
                    'decision': decision,
                    'reasons': reasons,
                    'confidence': ocr_result.get('confidence', 0.5),
                    'bounding_boxes': ocr_result.get('bounding_boxes', [])
                }
                
                if final_score > best_score:
                    best_result = result
                    best_score = final_score
        
        return best_result if best_result else {
            'registration_no': reg_numbers[0] if reg_numbers else None,
            'db_record': None,
            'ocr_extracted': {
                'raw_text': extracted_text,
                'name': None,
                'institution': None,
                'degree': None,
                'year': None
            },
            'field_scores': {},
            'final_score': 0.0,
            'decision': 'NOT_FOUND',
            'reasons': [f'Registration number {reg_numbers[0]} not found in database'],
            'confidence': ocr_result.get('confidence', 0.5)
        }
    
    def _extract_registration_numbers(self, text: str) -> List[str]:
        """Extract potential registration numbers from OCR text."""
        reg_numbers = []
        
        for pattern in self.reg_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and normalize the match
                clean_match = re.sub(r'USN:?\s*', '', match, flags=re.IGNORECASE)
                clean_match = re.sub(r'[-_\s]+', '', clean_match.upper())
                if clean_match not in reg_numbers and len(clean_match) > 3:
                    reg_numbers.append(clean_match)
        
        return reg_numbers
    
    def _lookup_registration(self, reg_no: str) -> Optional[Dict[str, Any]]:
        """Look up registration number in Supabase database."""
        try:
            # Try exact match first
            response = self.supabase.table('certificates') \
                .select('*') \
                .or_(f'reg_no.ilike.{reg_no},usn.ilike.{reg_no}') \
                .execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            # Try fuzzy matching
            all_records = self.supabase.table('certificates') \
                .select('reg_no, usn, id') \
                .execute()
            
            best_match = None
            best_score = 0
            
            for record in all_records.data:
                for field in ['reg_no', 'usn']:
                    if record.get(field):
                        score = fuzz.ratio(reg_no.upper(), record[field].upper()) / 100.0
                        if score > best_score and score > 0.8:
                            best_score = score
                            best_match = record['id']
            
            if best_match:
                response = self.supabase.table('certificates') \
                    .select('*') \
                    .eq('id', best_match) \
                    .execute()
                
                if response.data:
                    return response.data[0]
        
        except Exception as e:
            print(f"Supabase lookup error: {e}")
        
        return None
    
    def _extract_fields_from_ocr(self, text: str, db_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields from OCR text using the database record as a guide."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        words = text.split()
        
        extracted = {
            'raw_text': text,
            'name': None,
            'institution': None,
            'degree': None,
            'year': None,
            'father_name': None
        }
        
        # If NO database record, return empty (registration not found = reject)
        if not db_record:
            return extracted
        
        # Extract name using database guidance
        if db_record.get('name'):
            db_name = db_record['name'].upper()
            best_name_match = None
            best_name_score = 0
            
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    candidate = ' '.join(words[i:j]).upper()
                    if not any(skip in candidate for skip in ['CERTIFICATE', 'COMPLETION', 'CERTIFY']):
                        score = fuzz.ratio(candidate, db_name) / 100.0
                        if score > best_name_score and score > 0.6:
                            best_name_score = score
                            best_name_match = candidate
            
            extracted['name'] = best_name_match
        
        # Extract father's name
        if db_record.get('father_name'):
            db_father = db_record['father_name'].upper()
            best_father_score = 0
            best_father_match = None
            
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    candidate = ' '.join(words[i:j]).upper()
                    if not any(skip in candidate for skip in ['CERTIFICATE', 'UNIVERSITY']):
                        score = fuzz.ratio(candidate, db_father) / 100.0
                        if score > best_father_score and score > 0.6:
                            best_father_score = score
                            best_father_match = candidate
            
            extracted['father_name'] = best_father_match
        
        # Extract institution
        if db_record.get('institution'):
            db_institution = db_record['institution'].upper()
            best_inst_score = 0
            best_inst_match = None
            
            for line in lines:
                score = fuzz.partial_ratio(line.upper(), db_institution) / 100.0
                if score > best_inst_score and score > 0.7:
                    best_inst_score = score
                    best_inst_match = line
            
            extracted['institution'] = best_inst_match
        
        # Extract degree
        if db_record.get('degree'):
            db_degree = db_record['degree'].upper()
            
            for line in lines:
                score = fuzz.partial_ratio(line.upper(), db_degree) / 100.0
                if score > 0.7:
                    extracted['degree'] = line
                    break
        
        # Extract year
        if db_record.get('year'):
            db_year = db_record['year']
            if str(db_year) in text:
                extracted['year'] = db_year
            else:
                year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', text)
                if year_matches:
                    years = [int(y) for y in year_matches if 1990 <= int(y) <= 2030]
                    if years:
                        extracted['year'] = min(years, key=lambda x: abs(x - db_year))
        
        return extracted
    
    def _compare_fields(self, db_record: Dict[str, Any], 
                       ocr_extracted: Dict[str, Any]) -> Dict[str, float]:
        """Compare database record fields with OCR extracted fields."""
        scores = {}
        
        # Compare name
        if db_record.get('name') and ocr_extracted.get('name'):
            name_db = db_record['name'].upper().strip()
            name_ocr = ocr_extracted['name'].upper().strip()
            scores['name'] = max(
                fuzz.ratio(name_db, name_ocr) / 100.0,
                fuzz.token_sort_ratio(name_db, name_ocr) / 100.0
            )
        else:
            scores['name'] = 0.0
        
        # Compare father's name
        if db_record.get('father_name') and ocr_extracted.get('father_name'):
            father_db = db_record['father_name'].upper().strip()
            father_ocr = ocr_extracted['father_name'].upper().strip()
            scores['father_name'] = max(
                fuzz.ratio(father_db, father_ocr) / 100.0,
                fuzz.token_sort_ratio(father_db, father_ocr) / 100.0
            )
        else:
            scores['father_name'] = 0.0
        
        # Compare institution
        if db_record.get('institution') and ocr_extracted.get('institution'):
            inst_db = db_record['institution'].upper().strip()
            inst_ocr = ocr_extracted['institution'].upper().strip()
            scores['institution'] = fuzz.partial_ratio(inst_db, inst_ocr) / 100.0
        else:
            scores['institution'] = 0.0
        
        # Compare degree
        if db_record.get('degree') and ocr_extracted.get('degree'):
            degree_db = db_record['degree'].upper().strip()
            degree_ocr = ocr_extracted['degree'].upper().strip()
            scores['degree'] = fuzz.partial_ratio(degree_db, degree_ocr) / 100.0
        else:
            scores['degree'] = 0.0
        
        # Compare year
        if db_record.get('year') and ocr_extracted.get('year'):
            year_diff = abs(db_record['year'] - ocr_extracted['year'])
            if year_diff == 0:
                scores['year'] = 1.0
            elif year_diff == 1:
                scores['year'] = 0.9
            elif year_diff == 2:
                scores['year'] = 0.7
            else:
                scores['year'] = 0.0
        else:
            scores['year'] = 0.0
        
        return scores
    
    def _calculate_final_score(self, field_scores: Dict[str, float]) -> float:
        """Calculate weighted final score."""
        total_weight = sum(self.field_weights.values())
        weighted_sum = sum(
            score * self.field_weights.get(field, 0) 
            for field, score in field_scores.items()
        )
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _make_decision(self, final_score: float, field_scores: Dict[str, float], 
                      reg_no: str) -> Tuple[str, List[str]]:
        """Make final verification decision and provide reasons."""
        reasons = []
        
        for field, score in field_scores.items():
            if score >= 0.9:
                reasons.append(f"{field} match excellent ({score:.2f})")
            elif score >= 0.7:
                reasons.append(f"{field} match good ({score:.2f})")
            elif score >= 0.5:
                reasons.append(f"{field} match moderate ({score:.2f})")
            elif score > 0:
                reasons.append(f"{field} match poor ({score:.2f})")
            else:
                reasons.append(f"{field} not found or no match")
        
        reasons.append(f"Registration number {reg_no} found in database")
        
        if final_score >= self.authentic_threshold:
            decision = "AUTHENTIC"
            reasons.append(f"High confidence score ({final_score:.2f})")
        elif final_score >= self.suspect_threshold:
            decision = "SUSPECT"
            reasons.append(f"Moderate confidence score ({final_score:.2f}) - needs manual review")
        else:
            decision = "SUSPECT"
            reasons.append(f"Low confidence score ({final_score:.2f}) - likely fraudulent")
        
        return decision, reasons


# Backward compatibility wrapper
CertificateVerifier = SupabaseCertificateVerifier
