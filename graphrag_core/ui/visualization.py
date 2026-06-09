"""グラフ可視化関数

Neo4j公式 neo4j-viz ライブラリを使用したナレッジグラフの可視化。
"""

import streamlit as st


def get_node_type(node_name: str, node_label: str = None) -> str:
    """ノード名やラベルからタイプを推論"""
    if node_label and node_label != 'Unknown':
        return node_label

    # 人物判定
    person_keywords = ['太郎', '姫', '爺', '婆', '王', '侍', '人', '者']
    if any(kw in node_name for kw in person_keywords):
        return 'Person'

    # 場所判定
    place_keywords = ['山', '川', '島', '村', '城', '国', '都', '里']
    if any(kw in node_name for kw in place_keywords):
        return 'Place'

    # イベント判定
    event_keywords = ['戦', '旅', '退治', '発見', '誕生', '出会']
    if any(kw in node_name for kw in event_keywords):
        return 'Event'

    # 物判定
    object_keywords = ['宝', '刀', '船', '玉', '箱', '鏡']
    if any(kw in node_name for kw in object_keywords):
        return 'Object'

    return 'Other'


def get_color_for_type(node_type: str) -> str:
    """ノードタイプに応じた色を返す"""
    color_map = {
        'Person': '#FF6B6B',      # 赤系（人物）
        'Place': '#4ECDC4',       # 青緑系（場所）
        'Event': '#95E1D3',       # 緑系（イベント）
        'Object': '#FFE66D',      # 黄色系（物）
        'Organization': '#A8E6CF', # 薄緑（組織）
        'Other': '#95A5A6',       # グレー（その他）
        'Unknown': '#7F8C8D'      # 濃いグレー（不明）
    }
    return color_map.get(node_type, '#95A5A6')


def visualize_graph_neo4j_viz(graph_data):
    """Neo4j公式 neo4j-viz でグラフを可視化（HTMLを返す）"""
    try:
        from neo4j_viz import Node as VizNode, Relationship as VizRel, VisualizationGraph

        if not graph_data:
            st.warning("⚠️ グラフデータが空です（neo4j-viz）")
            return None

        node_dict = {}
        for item in graph_data:
            if 'source' not in item or 'target' not in item or 'relation' not in item:
                continue
            for key, type_key, deg_key in [
                ('source', 'source_type', 'source_degree'),
                ('target', 'target_type', 'target_degree'),
            ]:
                name = item[key]
                if name not in node_dict:
                    node_type = get_node_type(name, item.get(type_key))
                    node_dict[name] = {
                        'type': node_type,
                        'degree': item.get(deg_key, 1),
                        'color': get_color_for_type(node_type),
                    }

        nodes = [
            VizNode(
                id=name,
                caption=name,
                size=12 + min(info['degree'], 18),
                color=info['color'],
            )
            for name, info in node_dict.items()
        ]
        rels = [
            VizRel(source=item['source'], target=item['target'], caption=item['relation'])
            for item in graph_data
            if 'source' in item and 'target' in item and 'relation' in item
        ]
        if not nodes:
            st.warning("⚠️ neo4j-vizデータ不足: ノードが0個です")
            return None

        VG = VisualizationGraph(nodes=nodes, relationships=rels)
        html_obj = VG.render()
        return html_obj.data if hasattr(html_obj, 'data') else str(html_obj)

    except ImportError:
        st.info("ℹ️ neo4j-vizがインストールされていません（pip install neo4j-viz）")
        return None
    except Exception as e:
        st.warning(f"⚠️ neo4j-viz可視化エラー: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None


