from pathlib import Path
from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "er_diagram"

    dot = Digraph("ERDiagram", format="png")
    dot.attr(rankdir="LR", splines="ortho", nodesep="0.8", ranksep="1.0")
    dot.attr("node", shape="record", style="rounded,filled", fillcolor="lightyellow")

    dot.node(
        "users",
        "{users|"
        "id : INTEGER PK\\l"
        "username : STRING\\l"
        "password_hash : STRING\\l"
        "role : STRING\\l"
        "created_at : DATETIME\\l}"
    )

    dot.node(
        "uploads",
        "{uploads|"
        "id : INTEGER PK\\l"
        "user_id : INTEGER FK\\l"
        "filename : STRING\\l"
        "uploaded_at : DATETIME\\l}"
    )

    dot.node(
        "analyses",
        "{analyses|"
        "id : INTEGER PK\\l"
        "upload_id : INTEGER FK\\l"
        "profile : STRING\\l"
        "weights_json : TEXT\\l"
        "alpha : FLOAT\\l"
        "created_at : DATETIME\\l"
        "summary_json : TEXT\\l}"
    )

    dot.node(
        "profiles",
        "{profiles|"
        "id : INTEGER PK\\l"
        "name : STRING\\l"
        "created_at : DATETIME\\l}"
    )

    dot.node(
        "profile_criteria",
        "{profile_criteria|"
        "id : INTEGER PK\\l"
        "profile_id : INTEGER FK\\l"
        "code : STRING\\l"
        "label : STRING\\l"
        "weight : FLOAT\\l"
        "sort_order : INTEGER\\l"
        "is_active : BOOLEAN\\l}"
    )

    dot.edge("users", "uploads", label="1 : M")
    dot.edge("uploads", "analyses", label="1 : M")
    dot.edge("profiles", "profile_criteria", label="1 : M")

    dot.render(str(output_path), cleanup=True)
    print(f"ER-диаграмма сохранена: {output_path}.png")


if __name__ == "__main__":
    main()