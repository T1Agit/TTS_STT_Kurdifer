#!/usr/bin/env python3
"""
Test script to validate the training scripts structure and logic.
This doesn't require installing heavy dependencies like PyTorch, just checks the code structure.
"""

import sys
import ast
import os

def test_file_structure(filepath, expected_classes, expected_functions):
    """Test that a Python file has expected classes and functions"""
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Read and parse the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    
    # Extract classes and functions
    classes = []
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            if node.name != '__init__':  # Skip constructors
                functions.append(node.name)
    
    # Check expected classes
    print("\n📦 Classes:")
    all_classes_found = True
    for expected_class in expected_classes:
        if expected_class in classes:
            print(f"   ✅ {expected_class}")
        else:
            print(f"   ❌ {expected_class} (NOT FOUND)")
            all_classes_found = False
    
    # Check expected functions
    print("\n🔧 Functions:")
    all_functions_found = True
    for expected_func in expected_functions:
        if expected_func in functions:
            print(f"   ✅ {expected_func}")
        else:
            print(f"   ❌ {expected_func} (NOT FOUND)")
            all_functions_found = False
    
    # Check for main function
    has_main = 'main' in functions
    print(f"\n🚀 Entry point (main): {'✅' if has_main else '❌'}")
    
    success = all_classes_found and all_functions_found and has_main
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}")
    
    return success


def test_prepare_data():
    """Test prepare_data.py structure"""
    return test_file_structure(
        'prepare_data.py',
        expected_classes=['KurdishDataPreparation'],
        expected_functions=['main']
    )


def test_train_vits():
    """Test train_vits.py structure"""
    return test_file_structure(
        'train_vits.py',
        expected_classes=['KurdishTTSDataset', 'MMSFineTuner'],
        expected_functions=['main']
    )


def test_train_feedback():
    """Test train_feedback.py structure"""
    return test_file_structure(
        'train_feedback.py',
        expected_classes=['FeedbackDataManager'],
        expected_functions=['main']
    )


def test_imports():
    """Test that imports are structured correctly"""
    print(f"\n{'='*70}")
    print("Testing Import Statements")
    print(f"{'='*70}")
    
    files = ['prepare_data.py', 'train_vits.py', 'train_feedback.py']
    
    for filepath in files:
        print(f"\n📄 {filepath}:")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Count imports
        import_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
        
        print(f"   Import statements: {import_count}")
        
        # Check for key imports
        key_imports = {
            'prepare_data.py': ['soundfile', 'librosa', 'datasets'],
            'train_vits.py': ['torch', 'transformers', 'accelerate'],
            'train_feedback.py': ['soundfile', 'librosa']
        }
        
        file_key = os.path.basename(filepath)
        if file_key in key_imports:
            for key_import in key_imports[file_key]:
                if key_import in content:
                    print(f"   ✅ {key_import} imported")
                else:
                    print(f"   ⚠️  {key_import} not found")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Training Scripts Structure Validation")
    print("=" * 70)
    
    results = []
    
    # Test each script
    results.append(("prepare_data.py", test_prepare_data()))
    results.append(("train_vits.py", test_train_vits()))
    results.append(("train_feedback.py", test_train_feedback()))
    
    # Test imports
    test_imports()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 70)
        return 0
    else:
        print("❌ Some tests failed!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
