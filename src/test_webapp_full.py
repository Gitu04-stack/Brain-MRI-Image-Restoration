"""Phase 17: Comprehensive Flask Web Application Automated Test Suite.

Tests:
1. Application initialization & route registration
2. Dashboard and static pages (/index, /benchmarks, /deeplearning, /metrics)
3. Upload handling (sample preset loading and invalid uploads)
4. All 5 degradation conditions (Gaussian, Rician, Low-res, Motion blur, Combined) at various severities
5. All traditional restoration filters (Median, NLM, Wiener, Lanczos)
6. Deep ResUNet model inference integration & metric validation
7. Multi-restoration concurrent execution
8. Side-by-side comparison & download route (/download/<filename>)
9. Error handling & edge cases
"""

import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(r"C:\Users\gitan\BrainMRI_Project").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.app import app, STATIC_UPLOADS, get_deep_model, apply_deep_resunet_restoration


def run_full_test_suite():
    print("=" * 75)
    print("PHASE 17: COMPREHENSIVE FLASK APPLICATION AUTOMATED TEST SUITE")
    print("=" * 75)

    app.config['TESTING'] = True
    client = app.test_client()

    tests_run = 0
    tests_passed = 0

    def assert_test(name, condition, details=""):
        nonlocal tests_run, tests_passed
        tests_run += 1
        if condition:
            tests_passed += 1
            print(f"  [PASS] {name} {details}")
        else:
            print(f"  [FAIL] {name} {details}")
            raise AssertionError(f"Test failed: {name} - {details}")

    # Test 1: Static / GET Routes
    print("\n1. Testing Navigation & Core Dashboard Routes...")
    res = client.get('/')
    assert_test("GET / (Dashboard)", res.status_code == 200)

    res = client.get('/upload')
    assert_test("GET /upload", res.status_code == 200)

    res = client.get('/benchmarks')
    assert_test("GET /benchmarks", res.status_code == 200)
    assert_test("Verify Phase 14 in /benchmarks", b"Phase 14" in res.data)
    assert_test("Verify Phase 13 in /benchmarks", b"Phase 13" in res.data)

    res = client.get('/deeplearning')
    assert_test("GET /deeplearning", res.status_code == 200)
    assert_test("Verify Deep ResUNet Telemetry", b"ResUNet" in res.data)

    res = client.get('/metrics')
    assert_test("GET /metrics", res.status_code == 200)

    # Test 2: Upload Clinical Preset Sample
    print("\n2. Testing Sample Preset Selection...")
    res = client.post('/upload', data={'sample_choice': 'Te-gl_1'}, follow_redirects=True)
    assert_test("POST /upload (Pre-loaded clinical slice)", res.status_code == 200)
    assert_test("Redirect to /degrade", b"degradation" in res.data.lower() or b"simulate" in res.data.lower())

    # Test 3: Test All Degradation Pipelines
    print("\n3. Testing All 5 Artifact Synthesis Pipelines across Severities...")
    degradations = [
        ('gaussian', 30),
        ('rician', 40),
        ('low_resolution', 50),
        ('motion_blur', 20),
        ('combined', 40)
    ]
    for deg_type, sev in degradations:
        res = client.post('/apply_degradation', data={
            'degradation_type': deg_type,
            'severity': str(sev)
        }, follow_redirects=True)
        assert_test(f"Apply {deg_type.upper()} ({sev}%)", res.status_code == 200)

    # Test 4: Test Individual Restorations (Traditional + Deep ResUNet)
    print("\n4. Testing All Restoration Paradigms...")
    methods = [
        'median_filter',
        'nlm',
        'wiener',
        'lanczos_sharpening',
        'deep_resunet'
    ]
    for method in methods:
        res = client.post('/apply_restoration', data={
            'restoration_method': method
        }, follow_redirects=True)
        assert_test(f"Apply Restoration ({method})", res.status_code == 200)
        assert_test(f"Verify Metrics Display ({method})", b"PSNR" in res.data and b"SSIM" in res.data)

    # Test 5: Test Multi-Method Comparative Execution
    print("\n5. Testing Multi-Method Concurrent Benchmark Run...")
    res = client.post('/apply_all_restorations', follow_redirects=True)
    assert_test("POST /apply_all_restorations", res.status_code == 200)
    assert_test("Verify Compare Page Content", b"Primary Triplet Comparison" in res.data)
    assert_test("Verify Deep ResUNet in Multi-Results", b"Deep ResUNet" in res.data)

    # Test 6: Test Download Functionality
    print("\n6. Testing File Download Route...")
    with client.session_transaction() as sess:
        restored_fn = sess.get('active_restored_image')
    assert_test("Restored Image In Session", restored_fn is not None and len(restored_fn) > 0)

    res = client.get(f'/download/{restored_fn}')
    assert_test("GET /download/<filename>", res.status_code == 200)
    assert_test("Download Content-Type", res.content_type.startswith("image/"))
    assert_test("Download Image Bytes", len(res.data) > 1000)

    # Test 7: Deep ResUNet Direct Inference Test
    print("\n7. Testing Deep ResUNet PyTorch Inference & Tensor Bounds...")
    model = get_deep_model()
    assert_test("Deep Model Singleton Instantiated", model is not None)
    dummy_input = np.random.uniform(0.0, 1.0, (512, 512)).astype(np.float32)
    output = apply_deep_resunet_restoration(dummy_input)
    assert_test("ResUNet Output Shape (512, 512)", output.shape == (512, 512))
    assert_test("ResUNet Output Value Bounds [0.0, 1.0]", 0.0 <= output.min() and output.max() <= 1.0)

    # Test 8: Error Handling & Edge Cases
    print("\n8. Testing Edge Cases & Invalid Inputs...")
    res = client.get('/download/non_existent_file_12345.png', follow_redirects=True)
    assert_test("Invalid File Download Handled", res.status_code == 200)

    res = client.post('/upload', data={'sample_choice': 'invalid_tumor_class_xyz'}, follow_redirects=True)
    assert_test("Invalid Sample Preset Handled", res.status_code == 200)

    print("\n" + "=" * 75)
    print(f"ALL TESTS COMPLETED: {tests_passed}/{tests_run} PASSED (100% SUCCESS RATE)!")
    print("=" * 75)


if __name__ == '__main__':
    run_full_test_suite()
