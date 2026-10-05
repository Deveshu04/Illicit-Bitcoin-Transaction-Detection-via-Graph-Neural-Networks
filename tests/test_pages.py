import pytest

from helpers import strict_loads

MARK = '<script id="page-data" type="application/json">'


def page_data(html):
    start = html.index(MARK) + len(MARK)
    return strict_loads(html[start:html.index("</script>", start)])


@pytest.mark.parametrize("path", ["/", "/model"])
def test_pages_render_with_attribution(client, path):
    out = client.get(path)
    assert out.status_code == 200
    html = out.get_data(as_text=True)
    assert "CC BY-NC-ND 4.0" in html and "Elliptic Data Set" in html and "Weber et al." in html
    assert page_data(html)["headline"] == "hybrid"


def test_console_controls(client):
    html = client.get("/").get_data(as_text=True)
    for fragment in ('id="tx-input"', 'id="new-form"', 'id="graph"', 'id="drivers"', 'id="scores"', 'data-label="illicit"', 'data-label="licit"', 'data-label="unknown"', "/api/score"):
        assert fragment in html
    data = page_data(html)
    assert len(data["raw_names"]) == 165 and len(data["form_features"]) == 8 and data["max_links"] == 50


def test_model_page_tables(client):
    html = client.get("/model").get_data(as_text=True)
    for fragment in ("AUC-ROC", "Illicit F1", "0.962", "0.930", "Missed", "Met", 'id="roc"', 'id="per-step"', 'id="contrast"', "Cross-entropy, natural ratio", "22.0%"):
        assert fragment in html
    assert set(page_data(html)["roc"]) == {"hybrid", "graphsage", "raw_eng", "rf"}
