import os
import sys

# Ensure project root is in sys.path to import config.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


def validate_config():
    # Check 1: All directories in PATHS section can be created
    for key, path_val in config.PATHS.items():
        # If the path points to a file (like a JSON file), create its parent directory
        if path_val.endswith(".json"):
            dir_path = os.path.join(project_root, os.path.dirname(path_val))
        else:
            dir_path = os.path.join(project_root, path_val)
        os.makedirs(dir_path, exist_ok=True)
    print("PASS: All PATHS directories created")

    # Check 2: MODEL["node_feat_in"] == 9 and MODEL["edge_feat_in"] == 6
    assert config.MODEL["node_feat_in"] == 9, f"Expected node_feat_in to be 9, got {config.MODEL['node_feat_in']}"
    assert config.MODEL["edge_feat_in"] == 6, f"Expected edge_feat_in to be 6, got {config.MODEL['edge_feat_in']}"
    print("PASS: MODEL node_feat_in and edge_feat_in")

    # Check 3: MODEL["node_embed_dim"] == 128 and MODEL["gru_hidden_dim"] == 256
    assert config.MODEL["node_embed_dim"] == 128, f"Expected node_embed_dim to be 128, got {config.MODEL['node_embed_dim']}"
    assert config.MODEL["gru_hidden_dim"] == 256, f"Expected gru_hidden_dim to be 256, got {config.MODEL['gru_hidden_dim']}"
    print("PASS: MODEL node_embed_dim and gru_hidden_dim")

    # Check 4: TRAINING["lr"] == 1e-3 and TRAINING["max_epochs"] == 200
    assert config.TRAINING["lr"] == 1e-3, f"Expected lr to be 1e-3, got {config.TRAINING['lr']}"
    assert config.TRAINING["max_epochs"] == 200, f"Expected max_epochs to be 200, got {config.TRAINING['max_epochs']}"
    print("PASS: TRAINING lr and max_epochs")

    # Check 5: len(GRAPH["COUNTRY_LIST"]) == 44
    assert len(config.GRAPH["COUNTRY_LIST"]) == 44, f"Expected COUNTRY_LIST to have 44 elements, got {len(config.GRAPH['COUNTRY_LIST'])}"
    print("PASS: COUNTRY_LIST length")

    # Check 6: len(GRAPH["SECTOR_LIST"]) == 56
    assert len(config.GRAPH["SECTOR_LIST"]) == 56, f"Expected SECTOR_LIST to have 56 elements, got {len(config.GRAPH['SECTOR_LIST'])}"
    print("PASS: SECTOR_LIST length")

    # Check 7: len(EVENTS) == 6
    assert len(config.EVENTS) == 6, f"Expected EVENTS list length to be 6, got {len(config.EVENTS)}"
    print("PASS: EVENTS count")

    # Check 8: Each event dict has keys: name, date, hts_file, affected_importers, affected_exporters, description
    required_keys = {"name", "date", "hts_file", "affected_importers", "affected_exporters", "description"}
    for idx, event in enumerate(config.EVENTS):
        # We also support optional extra keys if any, but the required keys must be present
        for key in required_keys:
            assert key in event, f"Event {idx} ('{event.get('name', 'unknown')}') is missing key: '{key}'"
    print("PASS: Event dict keys")

    # Check 9: node_id(0, 0) == 0 and node_id(1, 0) == 56 and node_id(43, 55) == 2463
    val_0_0 = config.node_id(0, 0)
    val_1_0 = config.node_id(1, 0)
    val_43_55 = config.node_id(43, 55)
    assert val_0_0 == 0, f"Expected node_id(0, 0) to be 0, got {val_0_0}"
    assert val_1_0 == 56, f"Expected node_id(1, 0) to be 56, got {val_1_0}"
    assert val_43_55 == 2463, f"Expected node_id(43, 55) to be 2463, got {val_43_55}"
    print("PASS: node_id mapping checks")

    # Check 10: LEONTIEF["REG_EPS"] == 1e-4
    assert config.LEONTIEF["REG_EPS"] == 1e-4, f"Expected REG_EPS to be 1e-4, got {config.LEONTIEF['REG_EPS']}"
    print("PASS: LEONTIEF REG_EPS")

    print("All config checks passed.")


if __name__ == "__main__":
    validate_config()
