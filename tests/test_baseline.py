from cybersec_classifier import CyberTextClassifier


def test_classifier_trains_and_predicts():
    model = CyberTextClassifier(n_features=256).fit(
        ["remote code execution buffer overflow", "credential phishing email", "memory corruption exploit", "fake login page"],
        ["vulnerability", "phishing", "vulnerability", "phishing"],
    )
    assert model.predict(["buffer overflow"])[0] in {"vulnerability", "phishing"}

