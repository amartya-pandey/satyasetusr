import re
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from rapidfuzz import fuzz
import os

class CertificateVerifier:
    """Certificate verification engine using OCR results and database lookup."""
    
    def __init__(self, db_path: str = "certs.db"):
        """
        Initialize the verifier.
        
        Args:
            db_path: Path to SQLite database containing certificate records
        """
        self.db_path = db_path
        
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
        
        # Decision thresholds (more realistic for OCR scenarios)
        self.authentic_threshold = 0.75  # Lowered from 0.85
        self.suspect_threshold = 0.4     # Lowered from 0.5
    
    def verify_certificate(self, ocr_result: Dict[str, Any], 
                          image_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a certificate using OCR results and database lookup.
        
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
            # Step 2: Database lookup
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
    
    def _extract_fields_generic(self, text: str, lines: List[str], words: List[str]) -> Dict[str, Any]:
        """Extract fields using generic patterns (without database guidance)."""
        result = {}
        
        # Extract year using common patterns
        year_patterns = [
            r'(?:August|September|October|November|December|January|February|March|April|May|June|July)\s+(\d{4})',
            r'Year:?\s*(\d{4})',
            r'Datel?\s+\d+\s+(?:August|September|October|November|December|January|February|March|April|May|June|July)\s+(\d{4})',
            r'\b(20\d{2}|19\d{2})\b'
        ]
        
        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                year = int(match) if isinstance(match, str) else int(match[0])
                if 1990 <= year <= 2030 and year not in years_found:
                    years_found.append(year)
        
        if years_found:
            result['year'] = years_found[0]  # Use first found year
        
        # Extract institution - look for keywords
        institution_keywords = ['UNIVERSITY', 'COLLEGE', 'INSTITUTE', 'ACADEMY', 'SCHOOL', 'TECHNOLOGY']
        for line in lines:
            line_upper = line.upper()
            if any(keyword in line_upper for keyword in institution_keywords):
                # Check if line is substantial (not just keyword)
                if len(line) > 10:
                    result['institution'] = line
                    break
        
        # Extract degree - look for common patterns
        degree_patterns = {
            'BACHELOR': r'BACHELOR\s+(?:OF\s+)?(?:COMPUTER|BUSINESS|COMMERCE|SCIENCE|ARTS|TECHNOLOGY|ENGINEERING)',
            'BCA': r'\bBCA\b',
            'BBA': r'\bBBA\b',
            'BCOM': r'\bBCOM\b|B\.COM\b',
            'BSC': r'\bBSC\b|B\.SC\b',
            'BTECH': r'\bB\.?TECH\b',
            'BE': r'\bBE\b|B\.E\b',
            'MTECH': r'\bM\.?TECH\b',
            'MSC': r'\bMSC\b|M\.SC\b',
            'DIPLOMA': r'\bDIPLOMA\b'
        }
        
        for line in lines:
            for degree_name, pattern in degree_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    result['degree'] = line
                    break
            if result.get('degree'):
                break
        
        # Extract name - look for patterns like "Name of the Student" or names after common phrases
        name_patterns = [
            r'(?:Name\s+of\s+(?:the\s+)?Student|Student\s+Name|Name)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:Father(?:\'?s)?\s+Name|Mother(?:\'?s)?\s+Name)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:certify\s+that|this\s+is\s+to\s+certify\s+that)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['name'] = match.group(1).strip()
                break
        
        # Alternative name extraction: look for capitalized sequences
        if not result.get('name'):
            for i, line in enumerate(lines):
                # Skip first 2 lines (usually headers)
                if i < 2:
                    continue
                # Look for lines with multiple capitalized words (likely names)
                cap_words = re.findall(r'\b[A-Z][a-z]+\b', line)
                if 2 <= len(cap_words) <= 5:  # Typical name length
                    # Avoid institution names or common words
                    if not any(keyword in line.upper() for keyword in ['UNIVERSITY', 'COLLEGE', 'INSTITUTE', 'CERTIFICATE', 'COMPLETION']):
                        result['name'] = ' '.join(cap_words)
                        break
        
        return result
    
    def _extract_registration_numbers(self, text: str) -> List[str]:
        """Extract potential registration numbers from OCR text."""
        reg_numbers = []
        
        for pattern in self.reg_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and normalize the match
                clean_match = re.sub(r'USN:?\s*', '', match, flags=re.IGNORECASE)  # Remove USN: prefix
                clean_match = re.sub(r'[-_\s]+', '', clean_match.upper())
                if clean_match not in reg_numbers and len(clean_match) > 3:
                    reg_numbers.append(clean_match)
        
        # Also try to find the patterns with separators preserved
        for pattern in self.reg_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                normalized = re.sub(r'USN:?\s*', '', match, flags=re.IGNORECASE)  # Remove USN: prefix
                normalized = normalized.upper().strip()
                if normalized not in reg_numbers and len(normalized) > 3:
                    reg_numbers.append(normalized)
        
        return reg_numbers
    
    def _lookup_registration(self, reg_no: str) -> Optional[Dict[str, Any]]:
        """Look up registration number in database."""
        if not os.path.exists(self.db_path):
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try exact match first in both reg_no and usn columns
            cursor.execute("""
                SELECT reg_no, name, institution, degree, year, notes, father_name, usn
                FROM certificates 
                WHERE UPPER(reg_no) = ? OR UPPER(usn) = ?
            """, (reg_no.upper(), reg_no.upper()))
            
            result = cursor.fetchone()
            
            if not result:
                # Try fuzzy matching on both registration numbers and USNs
                cursor.execute("SELECT reg_no, usn FROM certificates")
                all_numbers = []
                for row in cursor.fetchall():
                    if row[0]:  # reg_no
                        all_numbers.append(row[0])
                    if row[1]:  # usn
                        all_numbers.append(row[1])
                
                best_match = None
                best_score = 0
                
                for db_number in all_numbers:
                    score = fuzz.ratio(reg_no.upper(), db_number.upper()) / 100.0
                    if score > best_score and score > 0.8:  # 80% similarity threshold
                        best_score = score
                        best_match = db_number
                
                if best_match:
                    cursor.execute("""
                        SELECT reg_no, name, institution, degree, year, notes, father_name, usn
                        FROM certificates 
                        WHERE reg_no = ? OR usn = ?
                    """, (best_match, best_match))
                    result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return {
                    'reg_no': result[0],
                    'name': result[1],
                    'institution': result[2],
                    'degree': result[3],
                    'year': result[4],
                    'notes': result[5],
                    'father_name': result[6],
                    'usn': result[7]
                }
        
        except Exception as e:
            print(f"Database lookup error: {e}")
        
        return None
    
    def _extract_fields_from_ocr(self, text: str, db_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields from OCR text using the database record as a guide."""
        
        # Clean text for easier matching
        clean_text = text.upper()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        words = text.split()
        
        extracted = {
            'raw_text': text,
            'name': None,
            'institution': None,
            'degree': None,
            'year': None
        }
        
        # If NO database record, return empty fields (registration not found = reject)
        if not db_record:
            extracted['raw_text'] = text
            return extracted
        
        # Database record exists - use it to guide extraction (PRIMARY METHOD)
        # Smart name extraction using fuzzy matching with database name
        if db_record.get('name'):
            db_name = db_record['name'].upper()
            db_name_parts = db_name.split()
            
            # Method 1: Look for name parts in the text
            best_name_match = None
            best_name_score = 0
            
            # Check all combinations of consecutive words
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):  # Check up to 3-word combinations
                    candidate = ' '.join(words[i:j]).upper()
                    # Remove common non-name words
                    if not any(skip in candidate for skip in ['CERTIFICATE', 'COMPLETION', 'CERTIFY', 'THAT', 'THIS', 'THE', 'FROM', 'YEAR', 'NUMBER']):
                        score = fuzz.ratio(candidate, db_name) / 100.0
                        if score > best_name_score and score > 0.6:  # At least 60% similarity
                            best_name_score = score
                            best_name_match = candidate
            
            # Method 2: Look for individual name parts
            if not best_name_match:
                found_parts = []
                for name_part in db_name_parts:
                    if len(name_part) > 2:  # Skip very short words like initials
                        for word in words:
                            if fuzz.ratio(word.upper(), name_part) > 0.8:
                                found_parts.append(word)
                                break
                
                if len(found_parts) >= len(db_name_parts) * 0.5:  # Found at least half the name parts
                    best_name_match = ' '.join(found_parts)
            
            extracted['name'] = best_name_match
        
        # Extract father's name if available in database
        if db_record.get('father_name'):
            db_father = db_record['father_name'].upper()
            
            # Look for father's name in text
            best_father_match = None
            best_father_score = 0
            
            for i in range(len(words)):
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    candidate = ' '.join(words[i:j]).upper()
                    if not any(skip in candidate for skip in ['CERTIFICATE', 'UNIVERSITY', 'COLLEGE', 'INSTITUTE']):
                        score = fuzz.ratio(candidate, db_father) / 100.0
                        if score > best_father_score and score > 0.6:
                            best_father_score = score
                            best_father_match = candidate
            
            if best_father_match:
                extracted['father_name'] = best_father_match
        
        # Smart institution extraction
        if db_record.get('institution'):
            db_institution = db_record['institution'].upper()
            
            # Method 1: Direct fuzzy matching with lines
            best_institution_match = None
            best_institution_score = 0
            
            for line in lines:
                score = fuzz.partial_ratio(line.upper(), db_institution) / 100.0
                if score > best_institution_score and score > 0.7:
                    best_institution_score = score
                    best_institution_match = line
            
            # Method 2: Look for institution keywords + fuzzy match
            if not best_institution_match:
                institution_keywords = ['UNIVERSITY', 'COLLEGE', 'INSTITUTE', 'ACADEMY', 'SCHOOL']
                for line in lines:
                    line_upper = line.upper()
                    if any(keyword in line_upper for keyword in institution_keywords):
                        score = fuzz.partial_ratio(line_upper, db_institution) / 100.0
                        if score > best_institution_score and score > 0.5:
                            best_institution_score = score
                            best_institution_match = line
            
            # Method 3: Look for key institution words in the database name
            if not best_institution_match:
                db_inst_words = db_institution.split()
                for db_word in db_inst_words:
                    if len(db_word) > 4:  # Skip short words like "THE", "OF"
                        for line in lines:
                            if db_word in line.upper():
                                extracted['institution'] = line
                                break
                        if extracted['institution']:
                            break
            
            if best_institution_match:
                extracted['institution'] = best_institution_match
        
        # Smart degree extraction
        if db_record.get('degree'):
            db_degree = db_record['degree'].upper()
            
            # Method 1: Direct fuzzy matching
            best_degree_match = None
            best_degree_score = 0
            
            for line in lines:
                score = fuzz.partial_ratio(line.upper(), db_degree) / 100.0
                if score > best_degree_score and score > 0.7:
                    best_degree_score = score
                    best_degree_match = line
            
            # Method 2: Look for degree abbreviations
            if not best_degree_match:
                degree_patterns = {
                    'BCA': r'\bBCA\b',
                    'BBA': r'\bBBA\b',
                    'BCOM': r'\bBCOM\b|B\.COM\b',
                    'BSC': r'\bBSC\b|B\.SC\b',
                    'BTECH': r'\bB\.?TECH\b',
                    'MTECH': r'\bM\.?TECH\b',
                    'MSC': r'\bMSC\b|M\.SC\b',
                    'PHD': r'\bPHD\b',
                    'DIPLOMA': r'\bDIPLOMA\b'
                }
                
                for line in lines:
                    line_upper = line.upper()
                    for degree_key, pattern in degree_patterns.items():
                        if re.search(pattern, line_upper):
                            if degree_key in db_degree or fuzz.partial_ratio(degree_key, db_degree) > 0.8:
                                best_degree_match = line
                                break
                    if best_degree_match:
                        break
            
            if best_degree_match:
                extracted['degree'] = best_degree_match
        
        # Smart year extraction - prefer the database year if found
        if db_record.get('year'):
            db_year = db_record['year']
            
            # Look for the exact year first
            if str(db_year) in text:
                extracted['year'] = db_year
            else:
                # Look for nearby years (±2 years tolerance)
                year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', text)
                if year_matches:
                    years = [int(y) for y in year_matches if 1990 <= int(y) <= 2030]
                    if years:
                        # Prefer years close to the database year
                        closest_year = min(years, key=lambda x: abs(x - db_year))
                        extracted['year'] = closest_year
        
        return extracted
    
    def _compare_fields(self, db_record: Dict[str, Any], 
                       ocr_extracted: Dict[str, Any]) -> Dict[str, float]:
        """Compare database record fields with OCR extracted fields."""
        
        scores = {}
        
        # Compare name - use multiple fuzzy matching methods
        if db_record.get('name') and ocr_extracted.get('name'):
            name_db = db_record['name'].upper().strip()
            name_ocr = ocr_extracted['name'].upper().strip()
            
            # Use the best of multiple matching algorithms
            ratio_score = fuzz.ratio(name_db, name_ocr) / 100.0
            partial_score = fuzz.partial_ratio(name_db, name_ocr) / 100.0
            token_sort_score = fuzz.token_sort_ratio(name_db, name_ocr) / 100.0
            
            scores['name'] = max(ratio_score, partial_score, token_sort_score)
        else:
            scores['name'] = 0.0
        
        # Compare father's name if available
        if db_record.get('father_name') and ocr_extracted.get('father_name'):
            father_db = db_record['father_name'].upper().strip()
            father_ocr = ocr_extracted['father_name'].upper().strip()
            
            ratio_score = fuzz.ratio(father_db, father_ocr) / 100.0
            partial_score = fuzz.partial_ratio(father_db, father_ocr) / 100.0
            token_sort_score = fuzz.token_sort_ratio(father_db, father_ocr) / 100.0
            
            scores['father_name'] = max(ratio_score, partial_score, token_sort_score)
        else:
            scores['father_name'] = 0.0
        
        # Compare institution - be more lenient with formatting
        if db_record.get('institution') and ocr_extracted.get('institution'):
            inst_db = db_record['institution'].upper().strip()
            inst_ocr = ocr_extracted['institution'].upper().strip()
            
            # Multiple comparison methods
            partial_score = fuzz.partial_ratio(inst_db, inst_ocr) / 100.0
            token_sort_score = fuzz.token_sort_ratio(inst_db, inst_ocr) / 100.0
            
            # Check if key institution words are present
            db_words = [w for w in inst_db.split() if len(w) > 3]
            word_match_score = 0
            if db_words:
                matched_words = sum(1 for word in db_words if word in inst_ocr)
                word_match_score = matched_words / len(db_words)
            
            scores['institution'] = max(partial_score, token_sort_score, word_match_score)
        else:
            scores['institution'] = 0.0
        
        # Compare degree - handle abbreviations and variations
        if db_record.get('degree') and ocr_extracted.get('degree'):
            degree_db = db_record['degree'].upper().strip()
            degree_ocr = ocr_extracted['degree'].upper().strip()
            
            # Direct comparison
            partial_score = fuzz.partial_ratio(degree_db, degree_ocr) / 100.0
            token_sort_score = fuzz.token_sort_ratio(degree_db, degree_ocr) / 100.0
            
            # Check for common degree abbreviations
            degree_mappings = {
                'BCA': ['BCA', 'BACHELOR', 'COMPUTER', 'APPLICATION'],
                'BBA': ['BBA', 'BACHELOR', 'BUSINESS', 'ADMINISTRATION'],
                'BCOM': ['BCOM', 'B.COM', 'BACHELOR', 'COMMERCE'],
                'BSC': ['BSC', 'B.SC', 'BACHELOR', 'SCIENCE'],
                'BTECH': ['BTECH', 'B.TECH', 'BACHELOR', 'TECHNOLOGY'],
                'MTECH': ['MTECH', 'M.TECH', 'MASTER', 'TECHNOLOGY'],
                'MSC': ['MSC', 'M.SC', 'MASTER', 'SCIENCE'],
                'PHD': ['PHD', 'DOCTOR', 'PHILOSOPHY'],
                'DIPLOMA': ['DIPLOMA']
            }
            
            # Check if degree keywords match
            keyword_score = 0
            for degree_key, keywords in degree_mappings.items():
                if any(kw in degree_db for kw in keywords):
                    if any(kw in degree_ocr for kw in keywords):
                        keyword_score = 0.9
                        break
            
            scores['degree'] = max(partial_score, token_sort_score, keyword_score)
        else:
            scores['degree'] = 0.0
        
        # Compare year - be more tolerant of nearby years
        if db_record.get('year') and ocr_extracted.get('year'):
            year_diff = abs(db_record['year'] - ocr_extracted['year'])
            if year_diff == 0:
                scores['year'] = 1.0
            elif year_diff == 1:
                scores['year'] = 0.9  # More tolerant
            elif year_diff == 2:
                scores['year'] = 0.7  # Still acceptable
            elif year_diff <= 3:
                scores['year'] = 0.5  # Moderate match
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
        
        # Analyze individual field scores
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
        
        # Make decision based on thresholds
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
