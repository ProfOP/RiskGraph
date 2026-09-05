from app.ml.inference import engine

def test_network_edges_reference_nodes_for_regression_customer():
    response = engine.network("CUST_48973")
    ids = {node["id"] for node in response["nodes"]}
    assert len(response["nodes"]) > 1
    assert len(response["edges"]) > 0
    assert all(edge["source"] in ids and edge["target"] in ids for edge in response["edges"])
    assert response["summary"]["relationship_count"] == len(response["edges"])
