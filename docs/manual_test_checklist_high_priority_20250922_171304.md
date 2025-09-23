
# Manual UI Test Checklist - Steel Cutting System
Generated: 2025-09-22 17:13:04
Priority Filter: HIGH

## Test Environment Setup
- [ ] Streamlit app running on localhost:8503
- [ ] Sample data files available (data0923.txt, sizaidata.txt, pi.txt)
- [ ] Material inventory populated
- [ ] Browser with developer tools available
- [ ] Network connectivity stable

## Test Execution

### CORE-001: Basic Panel Optimization Workflow
**Priority:** HIGH | **Time:** 5 min | **Category:** Core Functionality

**Test Steps:**
- [ ] 1. Navigate to main dashboard page
- [ ] 2. Click '🔧 切断最適化を開始 / Start Cutting Optimization'
- [ ] 3. Upload data0923.txt file using file uploader
- [ ] 4. Verify file upload success message appears
- [ ] 5. Check that panel data is parsed and displayed in table
- [ ] 6. Click 'Optimize' button
- [ ] 7. Wait for optimization to complete
- [ ] 8. Review optimization results

**Expected Results:**
- [ ] File uploads successfully without errors
- [ ] Panel data table shows parsed panels with PI codes
- [ ] Optimization completes within 30 seconds
- [ ] Results show multiple sheets with panel placements
- [ ] Efficiency metrics are displayed (should be >60%)
- [ ] Navigation to visualization page works

**Status:** [ ] PASS [ ] FAIL [ ] SKIP
**Notes:** _________________________________
**Issues Found:** _________________________


### CORE-002: Panel Placement Rate Validation (16/473 Issue)
**Priority:** HIGH | **Time:** 8 min | **Category:** Core Functionality

**Test Steps:**
- [ ] 1. Go to Cutting Optimization page
- [ ] 2. Upload data0923.txt (contains ~100+ panels)
- [ ] 3. Run optimization
- [ ] 4. Analyze placement results carefully
- [ ] 5. Check Analysis Results page
- [ ] 6. Review efficiency and placement statistics

**Expected Results:**
- [ ] Placement rate should be >90% (previously was 16/473 = 3.4%)
- [ ] Most panels should be successfully placed
- [ ] Material efficiency should be >60%
- [ ] No excessive number of empty sheets

**Status:** [ ] PASS [ ] FAIL [ ] SKIP
**Notes:** _________________________________
**Issues Found:** _________________________


### CORE-003: PI Code Expansion Functionality
**Priority:** HIGH | **Time:** 6 min | **Category:** Core Functionality

**Test Steps:**
- [ ] 1. Navigate to PI Management page
- [ ] 2. Verify PI code database is loaded
- [ ] 3. Check for PI code 77131000 (from sample data)
- [ ] 4. Go to Cutting Optimization
- [ ] 5. Upload data0923.txt
- [ ] 6. Examine parsed panel data
- [ ] 7. Look for expanded cutting dimensions

**Expected Results:**
- [ ] PI Management page shows loaded PI codes
- [ ] PI code 77131000 is found and has expansion data
- [ ] Parsed panels show both finished and cutting dimensions
- [ ] Cutting dimensions are larger than finished dimensions
- [ ] Expansion amounts are reasonable (typically 5-20mm)

**Status:** [ ] PASS [ ] FAIL [ ] SKIP
**Notes:** _________________________________
**Issues Found:** _________________________


### UI-004: Analysis Results Page Completeness
**Priority:** HIGH | **Time:** 6 min | **Category:** UI/UX Improvements

**Test Steps:**
- [ ] 1. Complete an optimization
- [ ] 2. Navigate to Analysis Results page
- [ ] 3. Check for efficiency metrics
- [ ] 4. Look for cost analysis
- [ ] 5. Verify material utilization data
- [ ] 6. Test any export functionality
- [ ] 7. Check for optimization summary

**Expected Results:**
- [ ] Page shows comprehensive optimization metrics
- [ ] Efficiency calculations are accurate
- [ ] Cost analysis is present and reasonable
- [ ] Material utilization is clearly displayed
- [ ] Summary information is complete
- [ ] Data export works if implemented

**Status:** [ ] PASS [ ] FAIL [ ] SKIP
**Notes:** _________________________________
**Issues Found:** _________________________


### UPLOAD-001: File Upload Prioritization
**Priority:** HIGH | **Time:** 7 min | **Category:** File Upload & Processing

**Test Steps:**
- [ ] 1. Go to Cutting Optimization page
- [ ] 2. Try uploading data0923.txt (large file)
- [ ] 3. Measure processing time
- [ ] 4. Try uploading a smaller test file
- [ ] 5. Compare processing times
- [ ] 6. Test multiple file upload scenarios
- [ ] 7. Check for processing priority indicators

**Expected Results:**
- [ ] Large files process within reasonable time (<30s)
- [ ] Processing time scales reasonably with file size
- [ ] Upload progress is indicated to user
- [ ] Processing prioritization works correctly
- [ ] No timeout or memory issues with large files

**Status:** [ ] PASS [ ] FAIL [ ] SKIP
**Notes:** _________________________________
**Issues Found:** _________________________


### UPLOAD-002: Format Detection and Validation
**Priority:** HIGH | **Time:** 8 min | **Category:** File Upload & Processing

**Test Steps:**
- [ ] 1. Upload data0923.txt (TSV manufacturing format)
- [ ] 2. Check format detection message
- [ ] 3. Create and upload a CSV version
- [ ] 4. Test with invalid data file
- [ ] 5. Try uploading non-data file (e.g., image)
- [ ] 6. Check error messages and handling

**Expected Results:**
- [ ] Format is correctly detected as 'manufacturing_tsv'
- [ ] CSV format is also detected correctly
- [ ] Invalid data generates appropriate error messages
- [ ] Non-data files are rejected gracefully
- [ ] Error messages are helpful and specific

**Status:** [ ] PASS [ ] FAIL [ ] SKIP
**Notes:** _________________________________
**Issues Found:** _________________________


### INT-001: End-to-End Workflow Integration
**Priority:** HIGH | **Time:** 12 min | **Category:** Integration Testing

**Test Steps:**
- [ ] 1. Start from dashboard
- [ ] 2. Upload cutting data file
- [ ] 3. Verify data parsing and PI expansion
- [ ] 4. Run optimization
- [ ] 5. View results and visualization
- [ ] 6. Navigate to Analysis Results
- [ ] 7. Check work instructions (if available)
- [ ] 8. Test export functionality

**Expected Results:**
- [ ] Complete workflow executes without errors
- [ ] Data flows correctly between components
- [ ] Each step produces expected results
- [ ] Navigation between steps is smooth
- [ ] Final output is complete and useful

**Status:** [ ] PASS [ ] FAIL [ ] SKIP
**Notes:** _________________________________
**Issues Found:** _________________________

