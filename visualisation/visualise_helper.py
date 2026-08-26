from collections import defaultdict
from dataclasses import asdict
import http.server
from pathlib import Path
import socketserver
import threading
import time
import webbrowser
import openpyxl
import json
import os
import re
import uuid
import json
from urllib.parse import quote

from tree import Tree

VIS_DIR = "visualisation"


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def extract_entail_scores(eval_text):
    match = re.findall(r"entail-1: (\d+(?:\.\d+)?), entail-2: (\d+(?:\.\d+)?)", eval_text)
    if match:
        return match[0]
    return ('-', '-')


def generate_json_from_excel(input_file, output_json=None):
    if output_json is None:
        output_json = os.path.join(VIS_DIR, 'tree_data.json')


    wb = openpyxl.load_workbook(input_file)
    sheet = wb.active
    data_rows = list(sheet.iter_rows(min_row=2, values_only=True))

    
    max_cols = max(len(r) for r in data_rows)
    prompt_cols = list(range(0, max_cols, 4))

    
    level_entries = []  
    for col in prompt_cols:
        entries = []
        for row in data_rows:
            text = str(row[col]) if col < len(row) and row[col] is not None else ""
            if not text or text.startswith("Error"):
                continue

            sim = safe_float(row[col + 3]) if col + 3 < len(row) else None
            e1, e2 = extract_entail_scores(str(row[col + 1])) if col + 1 < len(row) else ('-', '-')

            node = {
                'name': text[:100] + '...' if len(text) > 100 else text,
                'full_text': text,
                'similarity': sim,
                'entail1': e1,
                'entail2': e2,
                'children': []
            }
            entries.append(node)
        level_entries.append(entries)


    for lvl in range(1, len(level_entries)):
        parents = level_entries[lvl]
        children = level_entries[lvl - 1]
        for idx, parent in enumerate(parents):
            for child_idx in (2 * idx, 2 * idx + 1):
                if child_idx < len(children):
                    parent['children'].append(children[child_idx])

    roots = level_entries[-1]

    os.makedirs(VIS_DIR, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(roots, f, indent=2)

    print(f"[INFO] JSON tree generated successfully at: {output_json}")
    print(f"[INFO] Number of root nodes: {len(roots)}")


def generate_json_from_tree(src_json_path: Path) -> Path:
    """
    Convert the hierarchy stored in *10NNofA.json* (which now carries
    bidirectional links via .parent[] and .children[]) into the nested
    structure expected by the D3 viewer, and write it to *tree_data.json*.

    Each output node has:
        name, full_text, similarity, entail1, entail2, children[]
    """
    tree = Tree.load(src_json_path)

    # ---- quick helpers ----------------------------------------------------
    def short_name(node):
        text = node.prompt or f"id {node.id}"
        return (text[:120] + "…") if len(text) > 120 else text

    def entail_from_eval(ev: str):
        m = re.findall(r"\d+\.\d+", ev or "")
        return (float(m[0]), float(m[1])) if len(m) >= 2 else ("-", "-")

    # index for O(1) lookup
    by_id = {n.id: n for lvl in tree.levels for n in lvl.nodes}

    # recursive builder that re‑uses already‑built subtrees (handles multi‑parent)
    built: dict[int, dict] = {}

    def build(node_id: int) -> dict:
        if node_id in built:
            return built[node_id]

        n = by_id[node_id]
        ent1, ent2 = entail_from_eval(n.evaluation)

        node_dict = {
            "name": short_name(n),
            "full_text": n.prompt or "",
            "similarity": n.score,
            "entail1": ent1,
            "entail2": ent2,
            "children": []
        }
        built[node_id] = node_dict  # cache before recursion

        for cid in n.children:
            if cid in by_id:                 
                node_dict["children"].append(build(cid))

        return node_dict

    # roots: nodes with no parent links
    root_ids = [n.id for lvl in tree.levels for n in lvl.nodes if not n.parent]
    roots = [build(rid) for rid in root_ids]

    out_path = Path(__file__).parent / "tree_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(roots, fh, indent=2, ensure_ascii=False)

    print(f"[INFO] JSON tree written → {out_path}")
    print(f"[INFO] Roots: {len(roots)}")
    return out_path

def launch_tree_viewer(
    tree_json_path: Path,
    port: int = 8010,
    viewer_data_basename: str | None = None,
) -> None:
    """
    Serve the repo root via http.server, open viewer.html in a browser, and
    BLOCK until Ctrl+C. Self-contained: no separate server needed.
    The HTML loads *tree_data.json* relative to itself, or *viewer_data_basename*
    when given (?data=... in the URL).
    """
    import http.server, socketserver, webbrowser, os

    repo_root = Path(__file__).parent.parent  # serve from the repo root
    os.chdir(repo_root)

    handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True

    # find a free port, starting at the requested one
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = socketserver.TCPServer(("", p), handler)
            break
        except OSError:
            continue
    if httpd is None:
        raise OSError(f"no free port in {port}-{port + 19}")

    viewer_url = f"http://localhost:{p}/visualisation/viewer.html"
    if viewer_data_basename:
        # safe='/' keeps repo-relative paths (demo_trees/foo.json) intact
        viewer_url = f"{viewer_url}?data={quote(viewer_data_basename, safe='/')}"

    print(f"[INFO] Serving {repo_root} at http://localhost:{p}/")
    print(f"[INFO] Viewer: {viewer_url}")
    print("[INFO] Press Ctrl+C to stop.")
    webbrowser.open_new_tab(viewer_url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Viewer server stopped.")
    finally:
        httpd.server_close()

def generate_dag_json_from_tree(src_path: Path, out_path="visualisation/tree_data.json"):
    """
    Convert a multi-level JSON to flat DAG format with debug info.
    """
    import json
    import os

    if not os.path.exists(src_path):
        print(f"[ERROR] Source file not found: {src_path}")
        return

    with open(src_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    flat_nodes = []
    levels = tree_data.get("levels", [])
    
    print(f"\n--- DEBUG: Layer Analysis for {src_path.name} ---")
    print(f"Total levels found: {len(levels)}")

    for i, level in enumerate(levels):
        nodes_in_level = level.get("nodes", [])
        # Debug print for each layer
        print(f"Layer {i:2}: {len(nodes_in_level):4} nodes")
        
        for node in nodes_in_level:
            # BUG FIX: Ensure parent is a list and handle None/missing values
            raw_parents = node.get("parent", [])
            if raw_parents is None:
                p_list = []
            elif isinstance(raw_parents, (int, str)):
                p_list = [str(raw_parents)]
            else:
                p_list = [str(pid) for pid in raw_parents]

            flat_nodes.append({
                "id": str(node["id"]),
                "parentIds": p_list,
                "name": node.get("prompt") or node.get("name", ""),
                "full_text": node.get("prompt", ""),
                "similarity": node.get("score"),
                "summary_blocked": node.get("summary_blocked"),
                "evalScore": node.get("evalScore", "-"),
                "entail1": node.get("entail_1", "-"),
                "entail2": node.get("entail_2", "-"),
                "faithfulness1": node.get("faithfulness1", "-"),
                "faithfulness2": node.get("faithfulness2", "-"),
                "informativeness1": node.get("informativeness1", "-"),
                "informativeness2": node.get("informativeness2", "-"),
                "specificity_faithfulness": node.get("specificity_faithfulness", "-"),
                "specificity_informativeness": node.get("specificity_informativeness", "-"),
                "wb_score_mistral": node.get("wb_score_mistral", "-"),
                "wb_score_llama": node.get("wb_score_llama", "-"),
                "wb_scores": node.get("wb_scores", {})
            })

    # Final check for roots (nodes with no parents)
    roots = [n for n in flat_nodes if not n['parentIds']]
    print(f"Total roots identified: {len(roots)}")
    if len(roots) > 1:
        print(f"[WARNING] Multiple roots found! IDs: {[r['id'] for r in roots]}")
    print("-------------------------------------------\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(flat_nodes, f, indent=2)

    print(f"[✓] DAG JSON written to {out_path}")