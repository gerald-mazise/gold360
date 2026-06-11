def _make_conftest():
    return ""
for d in ["test_anomaly", "test_data", "test_features", "test_models",
           "test_pipeline", "test_policy", "test_validation", "test_weak_supervision"]:
    with open(f"tests/{d}/__init__.py", "w") as f:
        f.write("")
