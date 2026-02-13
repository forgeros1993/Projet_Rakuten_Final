# 🧪 Stratégie de Tests - Rakuten Product Classifier

## Executive Summary

Ce document définit une stratégie de tests **enterprise-grade** pour garantir la qualité, la fiabilité et la maintenabilité de l'application Rakuten Product Classifier.

**Objectif de couverture**: ≥ 85% du code critique

---

## 📊 Pyramide des Tests

```
                    ┌─────────────────┐
                    │   E2E Tests     │  ← 5% (Playwright)
                    │   (UI/Browser)  │
                    ├─────────────────┤
                    │  Integration    │  ← 20% (Streamlit + ML)
                    │     Tests       │
                    ├─────────────────┤
                    │   ML-Specific   │  ← 25% (Model Quality)
                    │     Tests       │
                    ├─────────────────┤
                    │                 │
                    │   Unit Tests    │  ← 50% (Functions/Classes)
                    │                 │
                    └─────────────────┘
```

---

## 🎯 1. TESTS UNITAIRES (Unit Tests)

### 1.1 Module `utils/mock_classifier.py`

| Test ID | Fonction testée | Description | Priorité |
|---------|----------------|-------------|----------|
| UT-MC-001 | `DemoClassifier.__init__` | Initialisation avec/sans config | HIGH |
| UT-MC-002 | `DemoClassifier.predict(text=...)` | Prédiction texte seul | HIGH |
| UT-MC-003 | `DemoClassifier.predict(image=...)` | Prédiction image seule | HIGH |
| UT-MC-004 | `DemoClassifier.predict(text, image)` | Prédiction multimodale | HIGH |
| UT-MC-005 | `ClassificationResult` | Dataclass attributes | MEDIUM |
| UT-MC-006 | `_generate_predictions` | Distribution probabilités | HIGH |
| UT-MC-007 | `ModelConfig` | Validation dataclass | MEDIUM |
| UT-MC-008 | `TEXT_MODELS` | 3 modèles texte présents | HIGH |
| UT-MC-009 | `IMAGE_MODELS` | 3 modèles image présents | HIGH |
| UT-MC-010 | `MultiModelClassifier` | Prédictions multi-modèles | HIGH |

**Assertions clés**:
```python
# UT-MC-002: Prédiction texte
def test_predict_text_returns_valid_result():
    clf = DemoClassifier()
    result = clf.predict(text="iPhone 15 Pro Max")

    assert result is not None
    assert result.category in CATEGORY_MAPPING.keys()
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.top_k_predictions) == 5
    assert sum(p[1] for p in result.top_k_predictions) <= 1.0
    assert result.source in ["mock_text", "text", "demo"]
```

### 1.2 Module `utils/category_mapping.py`

| Test ID | Fonction testée | Description | Priorité |
|---------|----------------|-------------|----------|
| UT-CM-001 | `CATEGORY_MAPPING` | 27 catégories présentes | HIGH |
| UT-CM-002 | `get_category_info(code)` | Retourne (name, full, emoji) | HIGH |
| UT-CM-003 | `get_category_info(invalid)` | Gestion code invalide | MEDIUM |
| UT-CM-004 | `get_category_emoji(code)` | Retourne emoji correct | LOW |
| UT-CM-005 | `get_all_categories()` | Retourne dict complet | MEDIUM |

**Assertions clés**:
```python
# UT-CM-001: Vérification 27 catégories
def test_category_mapping_has_27_categories():
    assert len(CATEGORY_MAPPING) == 27

# UT-CM-002: Structure correcte
def test_get_category_info_returns_tuple():
    name, full, emoji = get_category_info("2583")
    assert isinstance(name, str)
    assert isinstance(full, str)
    assert len(emoji) > 0  # Emoji présent
```

### 1.3 Module `utils/preprocessing.py`

| Test ID | Fonction testée | Description | Priorité |
|---------|----------------|-------------|----------|
| UT-PP-001 | `preprocess_product_text` | Nettoyage HTML | HIGH |
| UT-PP-002 | `preprocess_product_text` | Combinaison designation+description | HIGH |
| UT-PP-003 | `preprocess_product_text` | Gestion texte vide | MEDIUM |
| UT-PP-004 | `preprocess_product_text` | Caractères spéciaux | MEDIUM |
| UT-PP-005 | `validate_text_input` | Validation longueur | HIGH |
| UT-PP-006 | `clean_html` | Suppression balises | HIGH |

**Cas de test**:
```python
@pytest.mark.parametrize("input_text,expected_contains", [
    ("<p>Test</p>", "Test"),           # HTML removal
    ("  spaces  ", "spaces"),          # Trim
    ("UPPERCASE", "uppercase"),        # Lowercase (si applicable)
    ("", ""),                          # Empty string
    (None, ""),                        # None handling
])
def test_preprocess_handles_various_inputs(input_text, expected_contains):
    result = preprocess_product_text(input_text, "")
    if expected_contains:
        assert expected_contains in result.lower()
```

### 1.4 Module `utils/image_utils.py`

| Test ID | Fonction testée | Description | Priorité |
|---------|----------------|-------------|----------|
| UT-IU-001 | `load_image_from_upload` | Chargement JPEG | HIGH |
| UT-IU-002 | `load_image_from_upload` | Chargement PNG | HIGH |
| UT-IU-003 | `validate_image` | Image valide | HIGH |
| UT-IU-004 | `validate_image` | Image trop petite | MEDIUM |
| UT-IU-005 | `validate_image` | Image trop grande | MEDIUM |
| UT-IU-006 | `get_image_info` | Métadonnées correctes | LOW |
| UT-IU-007 | `resize_image` | Redimensionnement | MEDIUM |

### 1.5 Module `utils/data_loader.py`

| Test ID | Fonction testée | Description | Priorité |
|---------|----------------|-------------|----------|
| UT-DL-001 | `is_data_available` | Détection données réelles | HIGH |
| UT-DL-002 | `get_dataset_summary` | Retourne dict valide | HIGH |
| UT-DL-003 | `get_category_distribution` | DataFrame correct | HIGH |
| UT-DL-004 | `get_text_statistics` | Stats texte | MEDIUM |
| UT-DL-005 | `load_training_data` | Fallback mode démo | HIGH |

### 1.6 Module `config.py`

| Test ID | Fonction testée | Description | Priorité |
|---------|----------------|-------------|----------|
| UT-CF-001 | `APP_CONFIG` | Clés requises présentes | HIGH |
| UT-CF-002 | `MODEL_CONFIG` | Configuration modèle | HIGH |
| UT-CF-003 | `THEME` | Couleurs définies | LOW |
| UT-CF-004 | `ASSETS_DIR` | Path existe | MEDIUM |

---

## 🔗 2. TESTS D'INTÉGRATION (Integration Tests)

### 2.1 Tests de chargement des pages

| Test ID | Page testée | Description | Priorité |
|---------|-------------|-------------|----------|
| IT-PG-001 | `app.py` | Page accueil charge sans erreur | CRITICAL |
| IT-PG-002 | `1_📊_Données.py` | Page données charge | CRITICAL |
| IT-PG-003 | `2_⚙️_Preprocessing.py` | Page preprocessing charge | CRITICAL |
| IT-PG-004 | `3_🧠_Modèles.py` | Page modèles charge | CRITICAL |
| IT-PG-005 | `4_🔍_Démo.py` | Page démo charge | CRITICAL |
| IT-PG-006 | `5_📈_Performance.py` | Page performance charge | CRITICAL |
| IT-PG-007 | `6_💡_Conclusions.py` | Page conclusions charge | CRITICAL |

**Implémentation avec `streamlit.testing`**:
```python
from streamlit.testing.v1 import AppTest

def test_home_page_loads():
    """Test que la page d'accueil charge sans erreur."""
    at = AppTest.from_file("app.py")
    at.run(timeout=30)

    assert not at.exception
    assert "Rakuten" in at.markdown[0].value

def test_demo_page_classification_flow():
    """Test du flow complet de classification."""
    at = AppTest.from_file("pages/4_🔍_Démo.py")
    at.run()

    # Simuler saisie texte
    at.text_input[0].set_value("iPhone 15 Pro Max")
    at.text_area[0].set_value("Smartphone Apple dernière génération")
    at.button[0].click()
    at.run()

    assert not at.exception
    # Vérifier qu'un résultat est affiché
    assert any("Catégorie" in str(m.value) for m in at.markdown)
```

### 2.2 Tests de Session State

| Test ID | Description | Priorité |
|---------|-------------|----------|
| IT-SS-001 | Persistance du classifier entre pages | HIGH |
| IT-SS-002 | Historique des classifications | HIGH |
| IT-SS-003 | Sélection de modèle persiste | HIGH |
| IT-SS-004 | Reset session fonctionne | MEDIUM |

### 2.3 Tests de Navigation

| Test ID | Description | Priorité |
|---------|-------------|----------|
| IT-NV-001 | Navigation accueil → démo | HIGH |
| IT-NV-002 | Navigation accueil → données | HIGH |
| IT-NV-003 | Navigation démo → comparaison | HIGH |
| IT-NV-004 | Sidebar navigation | MEDIUM |

### 2.4 Tests d'intégration Classifier + UI

| Test ID | Description | Priorité |
|---------|-------------|----------|
| IT-CL-001 | Classification texte end-to-end | CRITICAL |
| IT-CL-002 | Classification image end-to-end | CRITICAL |
| IT-CL-003 | Changement de modèle | HIGH |
| IT-CL-004 | Comparaison 3 modèles simultanés | HIGH |

---

## 🤖 3. TESTS SPÉCIFIQUES ML (Machine Learning Tests)

### 3.1 Tests de Performance du Modèle (Model Quality Gates)

| Test ID | Métrique | Seuil minimum | Priorité |
|---------|----------|---------------|----------|
| ML-PF-001 | Accuracy globale | ≥ 75% | CRITICAL |
| ML-PF-002 | F1-Score macro | ≥ 70% | CRITICAL |
| ML-PF-003 | F1-Score weighted | ≥ 75% | HIGH |
| ML-PF-004 | Precision macro | ≥ 70% | HIGH |
| ML-PF-005 | Recall macro | ≥ 65% | HIGH |

**Implémentation**:
```python
import pytest
from sklearn.metrics import accuracy_score, f1_score, classification_report

class TestModelPerformance:
    """Tests de qualité du modèle avec seuils minimaux."""

    ACCURACY_THRESHOLD = 0.75
    F1_MACRO_THRESHOLD = 0.70

    @pytest.fixture(scope="class")
    def model_predictions(self, test_dataset):
        """Génère les prédictions sur le jeu de test."""
        clf = load_production_model()
        X_test, y_true = test_dataset
        y_pred = clf.predict(X_test)
        return y_true, y_pred

    def test_accuracy_above_threshold(self, model_predictions):
        """L'accuracy doit être ≥ 75%."""
        y_true, y_pred = model_predictions
        accuracy = accuracy_score(y_true, y_pred)

        assert accuracy >= self.ACCURACY_THRESHOLD, \
            f"Accuracy {accuracy:.2%} below threshold {self.ACCURACY_THRESHOLD:.2%}"

    def test_f1_macro_above_threshold(self, model_predictions):
        """Le F1-score macro doit être ≥ 70%."""
        y_true, y_pred = model_predictions
        f1 = f1_score(y_true, y_pred, average='macro')

        assert f1 >= self.F1_MACRO_THRESHOLD, \
            f"F1-macro {f1:.2%} below threshold {self.F1_MACRO_THRESHOLD:.2%}"
```

### 3.2 Tests de Non-Régression (Regression Tests)

| Test ID | Description | Priorité |
|---------|-------------|----------|
| ML-RG-001 | Accuracy ne baisse pas vs baseline | CRITICAL |
| ML-RG-002 | F1-score ne baisse pas | CRITICAL |
| ML-RG-003 | Prédictions identiques pour inputs fixes | HIGH |
| ML-RG-004 | Distribution des classes stable | MEDIUM |

**Implémentation avec snapshot**:
```python
import json
from pathlib import Path

BASELINE_FILE = Path("tests/baselines/model_baseline.json")

class TestModelRegression:
    """Tests de non-régression du modèle."""

    REGRESSION_TOLERANCE = 0.02  # 2% de tolérance

    @pytest.fixture
    def baseline_metrics(self):
        """Charge les métriques de référence."""
        with open(BASELINE_FILE) as f:
            return json.load(f)

    def test_no_accuracy_regression(self, current_metrics, baseline_metrics):
        """L'accuracy ne doit pas baisser de plus de 2%."""
        current = current_metrics["accuracy"]
        baseline = baseline_metrics["accuracy"]

        assert current >= baseline - self.REGRESSION_TOLERANCE, \
            f"Regression detected: {current:.2%} vs baseline {baseline:.2%}"

    def test_deterministic_predictions(self):
        """Mêmes inputs = mêmes outputs (déterminisme)."""
        clf = DemoClassifier(seed=42)

        input_text = "Console PlayStation 5"
        result1 = clf.predict(text=input_text)
        result2 = clf.predict(text=input_text)

        assert result1.category == result2.category
        assert abs(result1.confidence - result2.confidence) < 0.001
```

### 3.3 Tests de Robustesse (Robustness Tests)

| Test ID | Cas de test | Comportement attendu | Priorité |
|---------|-------------|---------------------|----------|
| ML-RB-001 | Texte vide | Retourne prédiction par défaut | HIGH |
| ML-RB-002 | Texte très long (10K chars) | Tronque et prédit | MEDIUM |
| ML-RB-003 | Caractères spéciaux/emojis | Nettoie et prédit | MEDIUM |
| ML-RB-004 | Image corrompue | Erreur gracieuse | HIGH |
| ML-RB-005 | Image très petite (10x10) | Erreur ou upscale | MEDIUM |
| ML-RB-006 | Image très grande (8000x8000) | Redimensionne | MEDIUM |
| ML-RB-007 | Texte multilingue | Traduit et prédit | HIGH |
| ML-RB-008 | HTML malicieux | Sanitize et prédit | HIGH |

**Implémentation**:
```python
class TestModelRobustness:
    """Tests de robustesse du modèle face aux edge cases."""

    @pytest.mark.parametrize("edge_case_text", [
        "",                              # Empty
        " " * 100,                       # Whitespace only
        "a" * 10000,                     # Very long
        "<script>alert('xss')</script>", # XSS attempt
        "🎮📱💻🖥️",                      # Emojis only
        "价格便宜质量好",                   # Chinese
        "Ценa хорошая",                  # Russian
        None,                            # None value
    ])
    def test_handles_edge_case_text(self, edge_case_text):
        """Le modèle gère les cas limites sans crash."""
        clf = DemoClassifier()

        # Ne doit pas lever d'exception
        try:
            result = clf.predict(text=edge_case_text or "")
            assert result is not None
            assert result.category in CATEGORY_MAPPING
        except ValueError:
            pass  # ValueError acceptable pour entrée invalide

    def test_handles_adversarial_input(self):
        """Test d'input adversarial."""
        clf = DemoClassifier()

        # Texte conçu pour confondre le modèle
        adversarial = "livre console telephone jardin piscine figurine"
        result = clf.predict(text=adversarial)

        # Doit quand même retourner une prédiction valide
        assert result.category in CATEGORY_MAPPING
        # Confiance devrait être plus basse
        assert result.confidence < 0.95
```

### 3.4 Tests de Consistance Inter-Modèles

| Test ID | Description | Priorité |
|---------|-------------|----------|
| ML-CS-001 | 3 modèles texte donnent top-3 similaires | MEDIUM |
| ML-CS-002 | 3 modèles image donnent top-3 similaires | MEDIUM |
| ML-CS-003 | Corrélation des confidences entre modèles | LOW |

### 3.5 Tests de Performance (Latency/Throughput)

| Test ID | Métrique | Seuil | Priorité |
|---------|----------|-------|----------|
| ML-LT-001 | Latence inférence texte | < 100ms | HIGH |
| ML-LT-002 | Latence inférence image | < 500ms | HIGH |
| ML-LT-003 | Throughput batch 100 | > 10/s | MEDIUM |
| ML-LT-004 | Mémoire max | < 2 GB | MEDIUM |

**Implémentation**:
```python
import time
import pytest

class TestInferencePerformance:
    """Tests de performance d'inférence."""

    MAX_TEXT_LATENCY_MS = 100
    MAX_IMAGE_LATENCY_MS = 500

    def test_text_inference_latency(self):
        """L'inférence texte doit être < 100ms."""
        clf = DemoClassifier()
        text = "Console PlayStation 5 nouvelle génération"

        # Warmup
        clf.predict(text=text)

        # Mesure
        start = time.perf_counter()
        for _ in range(100):
            clf.predict(text=text)
        elapsed = (time.perf_counter() - start) * 1000 / 100  # ms par inférence

        assert elapsed < self.MAX_TEXT_LATENCY_MS, \
            f"Text inference too slow: {elapsed:.1f}ms > {self.MAX_TEXT_LATENCY_MS}ms"

    @pytest.mark.slow
    def test_batch_throughput(self):
        """Le throughput batch doit être > 10 prédictions/seconde."""
        clf = DemoClassifier()
        texts = [f"Product {i}" for i in range(100)]

        start = time.perf_counter()
        for text in texts:
            clf.predict(text=text)
        elapsed = time.perf_counter() - start

        throughput = len(texts) / elapsed
        assert throughput > 10, f"Throughput too low: {throughput:.1f}/s"
```

### 3.6 Tests de Data Quality (Validation des données)

| Test ID | Description | Priorité |
|---------|-------------|----------|
| ML-DQ-001 | Pas de NaN dans features | CRITICAL |
| ML-DQ-002 | Toutes les catégories présentes | HIGH |
| ML-DQ-003 | Distribution des classes cohérente | MEDIUM |
| ML-DQ-004 | Pas de doublons dans test set | HIGH |
| ML-DQ-005 | Train/Test split correct (pas de fuite) | CRITICAL |

**Implémentation avec Great Expectations style**:
```python
class TestDataQuality:
    """Tests de qualité des données."""

    def test_no_nan_in_features(self, training_data):
        """Aucune valeur NaN dans les features."""
        X, y = training_data
        assert not X.isnull().any().any(), "NaN values found in features"

    def test_all_categories_represented(self, training_data):
        """Les 27 catégories sont représentées."""
        X, y = training_data
        unique_categories = y['prdtypecode'].unique()

        assert len(unique_categories) == 27, \
            f"Expected 27 categories, found {len(unique_categories)}"

    def test_no_data_leakage(self, train_data, test_data):
        """Pas de fuite de données train→test."""
        train_ids = set(train_data.index)
        test_ids = set(test_data.index)

        overlap = train_ids & test_ids
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} samples in both sets"
```

---

## 🌐 4. TESTS END-TO-END (E2E Tests)

### 4.1 Tests Playwright (Browser Automation)

| Test ID | Scénario | Priorité |
|---------|----------|----------|
| E2E-001 | Parcours complet: accueil → classification → résultat | CRITICAL |
| E2E-002 | Upload image et classification | HIGH |
| E2E-003 | Comparaison des 3 modèles | HIGH |
| E2E-004 | Navigation complète (6 pages) | HIGH |
| E2E-005 | Galerie d'exemples | MEDIUM |
| E2E-006 | Export CSV distribution | LOW |

**Implémentation**:
```python
from playwright.sync_api import Page, expect
import pytest

class TestE2EClassification:
    """Tests E2E avec Playwright."""

    BASE_URL = "http://localhost:8501"

    def test_full_classification_journey(self, page: Page):
        """Parcours utilisateur complet."""
        # 1. Page d'accueil
        page.goto(self.BASE_URL)
        expect(page.locator("text=Rakuten")).to_be_visible()

        # 2. Click sur "Classifier un Produit"
        page.click("button:has-text('Classifier')")
        page.wait_for_url("**/4_*Démo*")

        # 3. Remplir le formulaire
        page.fill("input[aria-label='Désignation']", "iPhone 15 Pro Max")
        page.fill("textarea[aria-label='Description']", "Smartphone Apple")

        # 4. Classifier
        page.click("button:has-text('Classifier le Produit')")

        # 5. Vérifier le résultat
        expect(page.locator(".result-card")).to_be_visible(timeout=10000)
        expect(page.locator("text=Catégorie")).to_be_visible()

    def test_model_comparison(self, page: Page):
        """Test comparaison des modèles."""
        page.goto(f"{self.BASE_URL}/3_🧠_Modèles")

        # Sélectionner mode texte
        page.click("text=Modèles Texte")

        # Remplir input
        page.fill("input", "Console PlayStation 5")

        # Lancer comparaison
        page.click("button:has-text('Comparer')")

        # Vérifier 3 résultats
        expect(page.locator(".model-card")).to_have_count(3)
```

### 4.2 Tests de Snapshot UI

```python
def test_home_page_snapshot(page: Page, assert_snapshot):
    """Snapshot de la page d'accueil."""
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    assert_snapshot(page.screenshot(), "home_page.png")
```

---

## 🔒 5. TESTS DE SÉCURITÉ (Security Tests)

| Test ID | Vulnérabilité | Test | Priorité |
|---------|--------------|------|----------|
| SEC-001 | XSS | Injection script dans input | HIGH |
| SEC-002 | Path Traversal | Upload fichier malicieux | HIGH |
| SEC-003 | DoS | Input très volumineux | MEDIUM |
| SEC-004 | Injection | Caractères spéciaux SQL-like | LOW |

```python
class TestSecurity:
    """Tests de sécurité."""

    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "javascript:alert('xss')",
        "<svg onload=alert('xss')>",
    ]

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_prevention(self, payload):
        """Les payloads XSS sont sanitizés."""
        result = preprocess_product_text(payload, "")

        assert "<script>" not in result
        assert "javascript:" not in result
        assert "onerror=" not in result
```

---

## 📁 6. STRUCTURE DES FICHIERS DE TEST

```
tests/
├── __init__.py
├── conftest.py                    # Fixtures partagées
├── pytest.ini                     # Configuration pytest
│
├── unit/                          # Tests unitaires
│   ├── __init__.py
│   ├── test_mock_classifier.py
│   ├── test_category_mapping.py
│   ├── test_preprocessing.py
│   ├── test_image_utils.py
│   ├── test_data_loader.py
│   └── test_config.py
│
├── integration/                   # Tests d'intégration
│   ├── __init__.py
│   ├── test_page_loading.py
│   ├── test_session_state.py
│   ├── test_navigation.py
│   └── test_classification_flow.py
│
├── ml/                           # Tests ML spécifiques
│   ├── __init__.py
│   ├── test_model_performance.py
│   ├── test_model_regression.py
│   ├── test_model_robustness.py
│   ├── test_inference_latency.py
│   └── test_data_quality.py
│
├── e2e/                          # Tests end-to-end
│   ├── __init__.py
│   ├── test_classification_journey.py
│   ├── test_model_comparison.py
│   └── snapshots/                # Screenshots de référence
│
├── security/                     # Tests de sécurité
│   ├── __init__.py
│   └── test_input_sanitization.py
│
├── baselines/                    # Données de référence
│   ├── model_baseline.json       # Métriques baseline
│   └── golden_predictions.json   # Prédictions de référence
│
└── fixtures/                     # Données de test
    ├── sample_images/
    │   ├── valid_product.jpg
    │   ├── corrupted.jpg
    │   └── too_small.png
    └── sample_texts/
        └── test_products.csv
```

---

## ⚙️ 7. CONFIGURATION

### 7.1 `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    unit: Unit tests (fast)
    integration: Integration tests
    ml: Machine Learning specific tests
    e2e: End-to-end browser tests
    slow: Slow tests (>5s)
    security: Security tests

# Options par défaut
addopts =
    -v
    --tb=short
    --strict-markers
    -ra
    --cov=utils
    --cov=pages
    --cov-report=html:coverage_report
    --cov-report=term-missing
    --cov-fail-under=80

# Timeout par test
timeout = 30

# Parallel execution
# addopts = -n auto  # Décommenter pour exécution parallèle
```

### 7.2 `conftest.py`

```python
"""
Fixtures partagées pour tous les tests.
"""
import pytest
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import io

# Ajouter le répertoire source au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "streamlit"))

from utils.mock_classifier import DemoClassifier, TEXT_MODELS, IMAGE_MODELS
from utils.category_mapping import CATEGORY_MAPPING


# =============================================================================
# FIXTURES CLASSIFIER
# =============================================================================
@pytest.fixture
def demo_classifier():
    """Retourne un classifier initialisé."""
    return DemoClassifier()


@pytest.fixture
def text_classifier():
    """Classifier configuré pour le texte."""
    from utils.mock_classifier import TEXT_MODELS
    config = TEXT_MODELS["camembert"]
    return DemoClassifier(model_config=config)


@pytest.fixture
def image_classifier():
    """Classifier configuré pour les images."""
    from utils.mock_classifier import IMAGE_MODELS
    config = IMAGE_MODELS["resnet50_svm"]
    return DemoClassifier(model_config=config)


# =============================================================================
# FIXTURES DONNÉES
# =============================================================================
@pytest.fixture
def sample_text():
    """Texte produit exemple."""
    return "Console PlayStation 5 nouvelle génération Sony"


@pytest.fixture
def sample_image():
    """Image produit exemple (100x100 rouge)."""
    img = Image.new('RGB', (100, 100), color='red')
    return img


@pytest.fixture
def sample_image_bytes():
    """Image en bytes pour simuler upload."""
    img = Image.new('RGB', (100, 100), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def all_category_codes():
    """Liste de tous les codes catégorie."""
    return list(CATEGORY_MAPPING.keys())


# =============================================================================
# FIXTURES DONNÉES ML
# =============================================================================
@pytest.fixture(scope="session")
def training_data():
    """Charge les données d'entraînement (cache session)."""
    # À adapter selon votre structure de données
    data_path = Path(__file__).parent.parent / "data"
    X_train = pd.read_csv(data_path / "X_train.csv")
    Y_train = pd.read_csv(data_path / "Y_train.csv")
    return X_train, Y_train


@pytest.fixture
def model_baseline():
    """Charge les métriques baseline."""
    baseline_path = Path(__file__).parent / "baselines" / "model_baseline.json"
    if baseline_path.exists():
        import json
        with open(baseline_path) as f:
            return json.load(f)
    return {
        "accuracy": 0.75,
        "f1_macro": 0.70,
        "f1_weighted": 0.75,
    }


# =============================================================================
# FIXTURES EDGE CASES
# =============================================================================
@pytest.fixture(params=[
    "",
    " " * 100,
    "a" * 10000,
    "<script>alert('xss')</script>",
    "🎮📱💻",
    None,
])
def edge_case_text(request):
    """Paramétrise les cas limites de texte."""
    return request.param


@pytest.fixture
def corrupted_image():
    """Retourne des bytes invalides (pas une image)."""
    return io.BytesIO(b"not an image content")


@pytest.fixture
def oversized_image():
    """Image très grande (8000x8000)."""
    img = Image.new('RGB', (8000, 8000), color='green')
    return img


# =============================================================================
# FIXTURES STREAMLIT
# =============================================================================
@pytest.fixture
def streamlit_app():
    """Initialise AppTest pour les tests Streamlit."""
    from streamlit.testing.v1 import AppTest
    return AppTest


# =============================================================================
# CONFIGURATION PLAYWRIGHT (E2E)
# =============================================================================
@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configuration du contexte navigateur."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "locale": "fr-FR",
    }


# =============================================================================
# HELPERS
# =============================================================================
def pytest_configure(config):
    """Configuration pytest au démarrage."""
    # Ajouter des markers personnalisés
    config.addinivalue_line("markers", "gpu: tests requiring GPU")
    config.addinivalue_line("markers", "real_model: tests with real ML model")
```

---

## 🚀 8. COMMANDES D'EXÉCUTION

### 8.1 Exécution par type

```bash
# Tous les tests
pytest

# Tests unitaires uniquement (rapides)
pytest -m unit

# Tests d'intégration
pytest -m integration

# Tests ML
pytest -m ml

# Tests E2E (nécessite serveur Streamlit)
pytest -m e2e

# Tests de sécurité
pytest -m security

# Exclure les tests lents
pytest -m "not slow"
```

### 8.2 Avec rapport de couverture

```bash
# HTML report
pytest --cov=. --cov-report=html

# Terminal report
pytest --cov=. --cov-report=term-missing

# Fail si couverture < 80%
pytest --cov=. --cov-fail-under=80
```

### 8.3 CI/CD Pipeline

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt

    - name: Run unit tests
      run: pytest -m unit --cov=. --cov-report=xml

    - name: Run integration tests
      run: pytest -m integration

    - name: Run ML tests
      run: pytest -m ml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

---

## 📊 9. MÉTRIQUES DE QUALITÉ

### Dashboard de Tests

| Métrique | Cible | Status |
|----------|-------|--------|
| Couverture code | ≥ 80% | 🔴 |
| Tests unitaires passants | 100% | 🔴 |
| Tests intégration passants | 100% | 🔴 |
| Tests ML passants | 100% | 🔴 |
| Temps d'exécution total | < 5 min | 🔴 |
| Accuracy modèle | ≥ 75% | 🔴 |
| F1-Score modèle | ≥ 70% | 🔴 |

### Badges README

```markdown
![Tests](https://github.com/xxx/rakuten-classifier/actions/workflows/tests.yml/badge.svg)
![Coverage](https://codecov.io/gh/xxx/rakuten-classifier/branch/main/graph/badge.svg)
![Model Accuracy](https://img.shields.io/badge/accuracy-85%25-green)
```

---

## 🎯 10. PRIORISATION D'IMPLÉMENTATION

### Phase 1: Foundation (Critique) - 2h
1. ✅ Structure des dossiers tests
2. ✅ `conftest.py` avec fixtures de base
3. ✅ Tests unitaires `mock_classifier.py`
4. ✅ Tests unitaires `category_mapping.py`

### Phase 2: Core (Haute priorité) - 3h
5. Tests unitaires `preprocessing.py`
6. Tests unitaires `image_utils.py`
7. Tests d'intégration chargement pages
8. Tests ML performance baseline

### Phase 3: Robustness (Moyenne priorité) - 2h
9. Tests de robustesse (edge cases)
10. Tests de non-régression ML
11. Tests de latence inférence

### Phase 4: E2E & Polish (Si temps) - 2h
12. Tests E2E Playwright
13. Tests de sécurité
14. Configuration CI/CD

---

## ✅ Checklist Soutenance

- [ ] ≥ 50 tests unitaires passants
- [ ] ≥ 10 tests d'intégration passants
- [ ] ≥ 5 tests ML spécifiques passants
- [ ] Couverture ≥ 80%
- [ ] Rapport HTML de couverture généré
- [ ] Temps d'exécution < 2 minutes
- [ ] Badge de tests dans README
- [ ] Démonstration live `pytest -v` pendant soutenance

---

*Document créé le 16 janvier 2025 - Stratégie de Tests Enterprise-Grade*
