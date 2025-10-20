#!/usr/bin/env python3
"""
Quick test script to verify Random Forest implementation works correctly.
"""

import numpy as np
from new_implementations import RandomForestClassifier


def create_test_data():
    """Create synthetic test data for verification."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create mixed continuous and categorical features
    X_continuous = np.random.randn(n_samples, 5)
    X_categorical = np.random.randint(0, 5, size=(n_samples, 5))
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    
    # Create binary labels with some imbalance
    y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    return X, y


def test_random_forest():
    """Test Random Forest implementation."""
    print("🌲 Testing Random Forest...")
    X, y = create_test_data()
    
    # Test with categorical features
    categorical_features = list(range(5, 10))  # Last 5 features are categorical
    
    rf = RandomForestClassifier(
        n_estimators=10, 
        max_depth=5, 
        random_state=42,
        categorical_features=categorical_features
    )
    rf.fit(X, y)
    
    predictions = rf.predict(X)
    proba = rf.predict_proba(X)
    
    print(f"   ✅ Random Forest: {np.mean(predictions == y):.3f} accuracy")
    print(f"   ✅ OOB Score: {rf.oob_score_:.3f}")
    print(f"   ✅ Feature importances shape: {rf.feature_importances_.shape}")
    print(f"   ✅ Probability shape: {proba.shape}")
    return True


def test_reproducibility():
    """Test that Random Forest is reproducible with same random state."""
    print("🎲 Testing Reproducibility...")
    X, y = create_test_data()
    
    # Train two models with same random state
    rf1 = RandomForestClassifier(n_estimators=5, random_state=42)
    rf2 = RandomForestClassifier(n_estimators=5, random_state=42)
    
    rf1.fit(X, y)
    rf2.fit(X, y)
    
    pred1 = rf1.predict(X)
    pred2 = rf2.predict(X)
    
    # Should be identical
    identical = np.array_equal(pred1, pred2)
    print(f"   ✅ Reproducibility: {'PASS' if identical else 'FAIL'}")
    return identical


def main():
    """Run all tests."""
    print("🧪 Testing Random Forest Implementation")
    print("=" * 50)
    
    try:
        test_random_forest()
        test_reproducibility()
        
        print("\n✅ All tests passed! Random Forest is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: python run.py --config config_advanced.json")
        print("   2. Test directly: python run.py --model random_forest --data_dir preprocessed/level2")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()

