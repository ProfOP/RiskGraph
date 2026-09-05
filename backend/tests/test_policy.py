from app.ml.inference import risk_band, recommend

def test_bands():
    assert risk_band(.19) == "LOW"
    assert risk_band(.2) == "MEDIUM"
    assert risk_band(.5) == "HIGH"
    assert risk_band(.8) == "CRITICAL"

def test_cost_policy():
    assert recommend(.1, 10000) == "ALLOW"
    assert recommend(.6, 499) == "SOFT_REVIEW"
    assert recommend(.6, 500) == "STEP_UP_VERIFICATION"
    assert recommend(.9, 1000) == "MANUAL_REVIEW"
